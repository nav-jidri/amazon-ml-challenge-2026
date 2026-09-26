from pathlib import Path
import csv


def load_ground_truth(path):
    ground_truth = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            matched = row["matched_entity_ids"].strip()

            if matched:
                ground_truth[s1_id] = {
                    x.strip()
                    for x in matched.split(",")
                    if x.strip()
                }
            else:
                ground_truth[s1_id] = set()

    return ground_truth


def load_candidates(path):
    candidates = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            candidate_text = row["candidate_entity_ids"].strip()

            if candidate_text:
                candidates[s1_id] = {
                    x.strip()
                    for x in candidate_text.split(",")
                    if x.strip()
                }
            else:
                candidates[s1_id] = set()

    return candidates


def evaluate(ground_truth, candidates):
    total_true_links = 0
    recovered_true_links = 0

    s1_total = len(ground_truth)
    s1_with_all_matches_recovered = 0
    s1_with_zero_candidates = 0

    for s1_id, true_matches in ground_truth.items():

        candidate_set = candidates.get(s1_id, set())

        total_true_links += len(true_matches)

        recovered = true_matches & candidate_set

        recovered_true_links += len(recovered)

        if not candidate_set:
            s1_with_zero_candidates += 1

        if true_matches and recovered == true_matches:
            s1_with_all_matches_recovered += 1

    candidate_link_recall = (
        recovered_true_links / total_true_links
        if total_true_links
        else 0.0
    )

    s1_entity_recall = (
        s1_with_all_matches_recovered / s1_total
        if s1_total
        else 0.0
    )

    return {
        "S1 entities": s1_total,
        "true links": total_true_links,
        "recovered true links": recovered_true_links,
        "candidate link recall": candidate_link_recall,
        "S1 with all true matches recovered":
            s1_with_all_matches_recovered,
        "S1 entity recall": s1_entity_recall,
        "S1 with zero candidates": s1_with_zero_candidates,
    }


if __name__ == "__main__":

    ground_truth_path = Path(
        "dataset-20260926T040315Z-1-001"
        "/dataset/train/train_ground_truth.tsv"
    )

    candidate_path = Path(
        "output/train_candidate_pairs.tsv"
    )

    print("Loading ground truth...")
    ground_truth = load_ground_truth(ground_truth_path)

    print("Loading candidate pairs...")
    candidates = load_candidates(candidate_path)

    print("Evaluating candidate recall...")

    results = evaluate(
        ground_truth,
        candidates
    )

    print("\n" + "=" * 60)
    print("P2 CANDIDATE RECALL")
    print("=" * 60)

    for key, value in results.items():

        if "recall" in key.lower():
            print(f"{key}: {value:.4%}")
        else:
            print(f"{key}: {value:,}")

    print("=" * 60)