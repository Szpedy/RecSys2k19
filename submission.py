from verify_submission import verify_subm
from score_submission import score_subm
from baseline_algorithm import rec_popular

submission_file = 'data/submission_popular.csv'
gt_file = '.data/groundTruth.csv'
test_file = 'data/test.csv'
data_path = "data"

if __name__ == '__main__':
    rec_popular.main(data_path)
    verify_subm.main(data_path, submission_file, test_file)
    score_subm.main(data_path, submission_file, gt_file)
