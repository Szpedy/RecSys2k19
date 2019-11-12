from src.verify_submission import verify_subm
from src.score_submission import score_subm
from src.baseline_algorithm import rec_popular

submission_file = '../data/submission_popular.csv'
ground_truth_file = '../data/groundTruth.csv'
test_file = '../data/test.csv'
data_path = '../data'

if __name__ == '__main__':
    # rec_popular.main()
    verify_subm.main()
    score_subm.main()
