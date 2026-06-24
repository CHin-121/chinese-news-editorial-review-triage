import unittest

from repro_pipeline.cli import build_parser


class CliParserTests(unittest.TestCase):
    def test_parser_accepts_calibration_only_options(self):
        args = build_parser().parse_args(
            [
                "--data",
                "data/raw/cnews.train.txt",
                "--calibration-only",
                "--calibration-config",
                "configs/calibration.json",
                "--figures-root",
                "results/figures",
            ]
        )

        self.assertTrue(args.calibration_only)
        self.assertEqual(args.calibration_config, "configs/calibration.json")
        self.assertEqual(args.figures_root, "results/figures")


if __name__ == "__main__":
    unittest.main()
