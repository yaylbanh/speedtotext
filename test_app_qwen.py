import unittest
from types import SimpleNamespace

import app_qwen


class QwenPipelineTests(unittest.TestCase):
    def test_chunk_plans_cover_long_speech_with_bounded_cores(self):
        plans = app_qwen._build_chunk_plans(
            [(0.0, 42.0), (42.3, 108.0)],
            total_duration=110.0,
            max_sec=50.0,
            overlap=1.0,
        )
        self.assertEqual(3, len(plans))
        self.assertTrue(all(p["core_end"] - p["core_start"] <= 50.0 for p in plans))
        self.assertEqual(0.0, plans[0]["audio_start"])
        self.assertGreater(plans[1]["core_start"], plans[0]["core_start"])

    def test_missing_speech_detects_large_hole(self):
        speech = [(0.0, 20.0)]
        units = [
            ("前", 0.2, 1.0),
            ("后", 17.0, 18.0),
        ]
        missing = app_qwen._find_missing_speech(speech, units, min_gap=1.5)
        self.assertTrue(any(start < 2.0 and end > 16.0 for start, end in missing))

    def test_context_keeps_defaults_and_adds_custom_terms(self):
        context = app_qwen._compose_context("顾长歌，太初圣地")
        self.assertIn("鸿蒙剑体", context)
        self.assertIn("顾长歌", context)

    def test_srt_grouping_stays_short(self):
        units = [
            (char, index * 0.2, index * 0.2 + 0.18)
            for index, char in enumerate("我是正道第一势力道宗的长老却和魔道女帝相爱")
        ]
        lines = app_qwen._group_units(units)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(text) <= app_qwen.MAX_CHARS for _, _, text in lines))
        self.assertTrue(all(end > start for start, end, _ in lines))

    def test_raw_transcript_punctuation_creates_semantic_breaks(self):
        sentence = "我是正道第一势力道宗的长老却和魔道女帝相爱正所谓正邪不两立"
        units = [
            (char, index * 0.15, index * 0.15 + 0.14)
            for index, char in enumerate(sentence)
        ]
        marked = app_qwen._apply_transcript_boundaries(
            units,
            "我是正道第一势力道宗的长老，却和魔道女帝相爱。正所谓正邪不两立。",
        )
        lines = app_qwen._group_units(marked)
        texts = [text for _, _, text in lines]
        self.assertEqual(
            [
                "我是正道第一势力道宗的长老",
                "却和魔道女帝相爱",
                "正所谓正邪不两立",
            ],
            texts,
        )

    def test_transcribe_plan_reads_raw_result_text(self):
        sentence = "我是长老却和女帝相爱"
        items = [
            SimpleNamespace(text=char, start_time=index * 0.2, end_time=index * 0.2 + 0.18)
            for index, char in enumerate(sentence)
        ]
        result = SimpleNamespace(
            text="我是长老，却和女帝相爱。",
            time_stamps=SimpleNamespace(items=items),
        )

        class FakeModel:
            def transcribe(self, *args, **kwargs):
                return [result]

        old_extract = app_qwen._extract_audio_chunk
        app_qwen._extract_audio_chunk = lambda *args, **kwargs: "fake.wav"
        try:
            units = app_qwen._transcribe_plan(
                FakeModel(),
                "audio.wav",
                {
                    "audio_start": 0.0,
                    "audio_end": 10.0,
                    "core_start": 0.0,
                    "core_end": 10.0,
                },
                "",
                ".",
                "test",
            )
        finally:
            app_qwen._extract_audio_chunk = old_extract

        lines = app_qwen._group_units(units)
        self.assertEqual(["我是长老", "却和女帝相爱"], [text for _, _, text in lines])

    def test_batch_transcription_preserves_plan_order(self):
        plans = [
            {"audio_start": 0.0, "audio_end": 5.0, "core_start": 0.0, "core_end": 5.0},
            {"audio_start": 5.0, "audio_end": 10.0, "core_start": 5.0, "core_end": 10.0},
        ]
        results = [
            SimpleNamespace(
                text="第一。",
                time_stamps=SimpleNamespace(
                    items=[SimpleNamespace(text="第一", start_time=0.1, end_time=1.0)]
                ),
            ),
            SimpleNamespace(
                text="第二。",
                time_stamps=SimpleNamespace(
                    items=[SimpleNamespace(text="第二", start_time=0.2, end_time=1.1)]
                ),
            ),
        ]

        class FakeBatchModel:
            def transcribe(self, audio, context, language, return_time_stamps):
                self.batch_size = len(audio)
                return results

        model = FakeBatchModel()
        units = app_qwen._transcribe_batch_inputs(
            model,
            [("audio-1", 16000), ("audio-2", 16000)],
            plans,
            "context",
        )
        self.assertEqual(2, model.batch_size)
        self.assertEqual(["第一", "第二"], [app_qwen._plain_text(x[0]) for x in units])
        self.assertAlmostEqual(5.2, units[1][1])

    def test_batch_failure_falls_back_to_single_items(self):
        plans = [
            {"audio_start": 0.0, "audio_end": 5.0, "core_start": 0.0, "core_end": 5.0},
            {"audio_start": 5.0, "audio_end": 10.0, "core_start": 5.0, "core_end": 10.0},
        ]

        class FallbackModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, audio, context, language, return_time_stamps):
                self.calls.append(len(audio))
                if len(audio) > 1:
                    raise RuntimeError("out of memory")
                text = "甲" if audio[0][0] == "audio-1" else "乙"
                return [
                    SimpleNamespace(
                        text=text + "。",
                        time_stamps=SimpleNamespace(
                            items=[SimpleNamespace(text=text, start_time=0.1, end_time=0.5)]
                        ),
                    )
                ]

        model = FallbackModel()
        units = app_qwen._transcribe_batch_inputs(
            model,
            [("audio-1", 16000), ("audio-2", 16000)],
            plans,
            "context",
        )
        self.assertEqual([2, 1, 1], model.calls)
        self.assertEqual(["甲", "乙"], [app_qwen._plain_text(x[0]) for x in units])


if __name__ == "__main__":
    unittest.main()
