import unittest
import sys
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

    def test_long_narration_breaks_on_story_transitions(self):
        sentence = "到那时就可以正大光明的和女帝在一起就在我思索之际"
        units = [
            (char, index * 0.12, index * 0.12 + 0.1)
            for index, char in enumerate(sentence)
        ]
        lines = app_qwen._group_units(units)
        self.assertEqual(
            [
                "到那时",
                "就可以正大光明的和女帝在一起",
                "就在我思索之际",
            ],
            [text for _, _, text in lines],
        )

    def test_smart_wrap_avoids_breaking_domain_terms(self):
        sentence = "而且鸿蒙剑体的觉醒让我的剑道修为大幅提升"
        units = [
            (char, index * 0.12, index * 0.12 + 0.1)
            for index, char in enumerate(sentence)
        ]
        lines = app_qwen._group_units(units)
        texts = [text for _, _, text in lines]
        self.assertEqual(
            ["而且鸿蒙剑体的觉醒", "让我的剑道修为大幅提升"],
            texts,
        )
        self.assertTrue(any("鸿蒙剑体" in text for text in texts))

    def test_smart_wrap_balances_unpunctuated_long_clause(self):
        sentence = "一只由灵气凝聚而成的小仙鹤轻盈的落到了我的手中"
        units = [
            (char, index * 0.12, index * 0.12 + 0.1)
            for index, char in enumerate(sentence)
        ]
        lines = app_qwen._group_units(units)
        lengths = [len(text) for _, _, text in lines]
        self.assertTrue(all(length <= app_qwen.MAX_CHARS for length in lengths))
        self.assertLessEqual(max(lengths) - min(lengths), 5)

    def test_word_boundaries_prevent_splitting_compound_words(self):
        class FakeJieba:
            @staticmethod
            def setLogLevel(level):
                return None

            @staticmethod
            def add_word(word, freq):
                return None

            @staticmethod
            def lcut(text, cut_all=False, HMM=True):
                return ["我是", "正道", "第一", "势力", "道宗", "的", "长老"]

        old_module = sys.modules.get("jieba")
        old_ready = app_qwen._JIEBA_READY
        sys.modules["jieba"] = FakeJieba
        app_qwen._JIEBA_READY = False
        try:
            boundaries = app_qwen._word_boundary_offsets("我是正道第一势力道宗的长老")
        finally:
            app_qwen._JIEBA_READY = old_ready
            if old_module is None:
                del sys.modules["jieba"]
            else:
                sys.modules["jieba"] = old_module

        # "道宗" nam o offset 8-10, nen khong co boundary o giua offset 9.
        self.assertNotIn(9, boundaries)
        self.assertIn(10, boundaries)

    def test_common_narration_phrases_create_clean_breaks(self):
        sentence = "我嘴角不自觉的露出一抹温柔的笑容脑海中也浮现出那个女孩"
        units = [
            (char, index * 0.12, index * 0.12 + 0.1)
            for index, char in enumerate(sentence)
        ]
        lines = app_qwen._group_units(units)
        texts = [text for _, _, text in lines]
        self.assertEqual("我嘴角", texts[0])
        self.assertTrue(any(text.startswith("脑海中") for text in texts))

    def test_protected_character_title_stays_together(self):
        sentence = "此刻全都恭敬的跪拜在一位紫衣女子身前"
        units = [
            (char, index * 0.12, index * 0.12 + 0.1)
            for index, char in enumerate(sentence)
        ]
        lines = app_qwen._group_units(units)
        texts = [text for _, _, text in lines]
        self.assertTrue(any("紫衣女子" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
