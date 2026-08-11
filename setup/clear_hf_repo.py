import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Example: Yaraan/yaraan-hunza-asr-comparison-results")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--keep", action="append", default=[".gitattributes"])
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set.")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type=args.repo_type, exist_ok=True)
    files = api.list_repo_files(args.repo_id, repo_type=args.repo_type)
    files_to_delete = [path for path in files if path not in set(args.keep)]

    if not files_to_delete:
        print(f"{args.repo_id} is already clean.")
        return

    for path in files_to_delete:
        print(f"Deleting {path}")
        api.delete_file(
            path_in_repo=path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Remove old result file: {path}",
        )

    print(f"Cleaned {len(files_to_delete)} files from {args.repo_id}.")


if __name__ == "__main__":
    main()
