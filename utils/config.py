from utils.args import Args

# Configurations
config = Args({'num_nodes': 1,
               'cpus': 1,
               'gpus': 1,
               'workers': 4,
               'model_path': "../weights"
               })

# Configuration RAY
resources_per_trial = Args({"cpus": 3,
                            "gpus": 1})

# -----------------------------------------------------------------------------
