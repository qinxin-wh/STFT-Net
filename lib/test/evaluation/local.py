from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.network_path = ''    # Where tracking networks are stored.
    settings.otb_path = ''
    settings.prj_dir = ''
    settings.result_plot_path = ''
    settings.results_path = ''    # Where to store tracking results
    settings.save_dir = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

