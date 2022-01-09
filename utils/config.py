from box import Box

# Configurations
config = Box({'num_nodes': 1,
               'cpus': 1,
               'gpus': 1,
               'workers': 4,
               'model_path': "../weights"
               })

# Configuration RAY
resources_per_trial = Box({"cpus": 3,
                            "gpus": 1})

# -----------------------------------------------------------------------------
