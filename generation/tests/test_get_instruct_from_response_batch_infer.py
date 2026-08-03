import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import get_instruct_from_response_batch_infer as extractor


class RestoreBatchCommitTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.commit_file = self.root / "commits.jsonl"
        self.oneshot_file = self.root / "oneshot.jsonl"
        self.oneshot_file.write_text(
            json.dumps({"code_before": "example"}) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_commit_records(self):
        records = [
            {
                "commit": "invalid",
                "old_contents": "invalid-1\ninvalid-2\ninvalid-3",
                "new_contents": "",
            },
            {
                "commit": "short",
                "old_contents": "only-one-line",
                "new_contents": "new",
            },
            {
                "commit": "commit-a",
                "old_contents": "\n".join(f"a-{index}" for index in range(5)),
                "new_contents": "new-a",
            },
            {
                "commit": "commit-b",
                "old_contents": "\n".join(f"b-{index}" for index in range(5)),
                "new_contents": "new-b",
            },
        ]
        self.commit_file.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def replay(self, count=5):
        return extractor.replay_prompt_metadata(
            count,
            commit_input_file=self.commit_file,
            oneshot_input_file=self.oneshot_file,
            random_seed=42,
            min_snippet_lines=2,
            max_snippet_lines=3,
        )

    def test_replay_skips_invalid_and_short_records(self):
        self.write_commit_records()

        metadata = self.replay()

        self.assertEqual(list(metadata), [1, 2, 3, 4, 5])
        for item in metadata.values():
            self.assertEqual(len(item["commit"]), 2)
            self.assertEqual(set(item["commit"]), {"commit-a", "commit-b"})
            self.assertTrue(
                all(
                    2 <= len(snippet.splitlines()) <= 3
                    for snippet in item["code_snippet"]
                )
            )

    def test_restores_non_initial_request_after_full_replay(self):
        self.write_commit_records()
        metadata = self.replay(count=3)
        request = {
            "code_snippet": [
                f"\n{snippet}\n" for snippet in metadata[3]["code_snippet"]
            ],
            "commit": "",
        }
        requests = {3: request}

        extractor.restore_request_commits(requests, metadata)

        self.assertEqual(request["commit"], metadata[3]["commit"])
        self.assertTrue(all(isinstance(item, str) for item in request["commit"]))

    def test_rejects_replayed_snippet_mismatch(self):
        self.write_commit_records()
        metadata = self.replay(count=1)
        requests = {
            1: {
                "code_snippet": ["different", "content"],
                "commit": "",
            }
        }

        with self.assertRaisesRegex(
            extractor.BatchResultError,
            "Replayed code snippets do not match.*request-1",
        ):
            extractor.restore_request_commits(requests, metadata)

    def test_rejects_invalid_sampling_limits(self):
        self.write_commit_records()

        with self.assertRaisesRegex(
            extractor.BatchResultError,
            "max_snippet_lines",
        ):
            extractor.replay_prompt_metadata(
                1,
                commit_input_file=self.commit_file,
                oneshot_input_file=self.oneshot_file,
                min_snippet_lines=3,
                max_snippet_lines=2,
            )


if __name__ == "__main__":
    unittest.main()
