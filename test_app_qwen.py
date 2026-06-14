import unittest

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


if __name__ == "__main__":
    unittest.main()
