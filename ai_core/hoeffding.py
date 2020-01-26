from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

stream = FileStream("../kddcup99.csv")

stream.prepare_for_use()

# 2. Instantiate the HoeffdingTree classifier
ht = HoeffdingTree()

# 3. Setup the evaluator
#(n_wait=200, max_samples=100000, batch_size=1, pretrain_size=200, max_time=inf, metrics=None, 
#output_file=None, show_plot=False, restart_stream=True, data_points_for_classification=False)
evaluator = EvaluatePrequential(show_plot=False,
                                pretrain_size=200,
                                max_samples=300,
                                metrics=['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                output_file='aa.csv')
# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ht)
