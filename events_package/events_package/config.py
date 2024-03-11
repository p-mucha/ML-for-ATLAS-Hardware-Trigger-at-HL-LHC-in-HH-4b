config_five_layers = {
    "layers": ["psb", "emb1", "emb2", "emb3", "hab1"],
    "noise_thresholds": {"psb": 100, "emb1": 50, "emb2": 100, "emb3": 50, "hab1": 100},
    "dimensions": {
        "psb": (7, 9),
        "emb1": (3, 17),
        "emb2": (7, 9),
        "emb3": (7, 9),
        "hab1": (7, 9),
    },
    "columns": [
        "event_no",
        "candidate_no",
        "z",
        "et",
        "eta",
        "phi",
        "psb_eta",
        "psb_phi",
        "emb1_eta",
        "emb1_phi",
        "emb2_eta",
        "emb2_phi",
        "emb3_eta",
        "emb3_phi",
        "hab1_eta",
        "hab1_phi",
    ],
    "eta_granularity": {
        "psb": 0.0251791,
        "emb1": 0.0031264,
        "emb2": 0.0251791,
        "emb3": 0.05035,
        "hab1": 0.1007168,
    },
    "phi_granularity": {
        "psb": 0.098174,
        "emb1": 0.098174,
        "emb2": 0.0245437,
        "emb3": 0.0245437,
        "hab1": 0.098174,
    },
}

config_jet_candidate = {
    "layers": ["psb", "emb1", "emb2", "emb3", "hab1", "hab2", "hab3"],
    "noise_thresholds": {
        "psb": 100,
        "emb1": 50,
        "emb2": 100,
        "emb3": 50,
        "hab1": 100,
        "hab2": 100,
        "hab3": 100,
    },
    "dimensions": {
        "psb": (3, 9),
        "emb1": (3, 65),
        "emb2": (9, 9),
        "emb3": (9, 5),
        "hab1": (3, 3),
        "hab2": (3, 3),
        "hab3": (3, 5),
    },
    "columns": [
        "event_no",
        "candidate_no",
        "matched_truth_no",
        "truth_pdgID",
        "matched_topo_no",
        "placeholder",
        "z",
        "et",
        "eta",
        "phi",
        "matched_topo_et",
        "matched_topo_eta",
        "matched_topo_phi",
        "psb_eta",
        "psb_phi",
        "emb1_eta",
        "emb1_phi",
        "emb2_eta",
        "emb2_phi",
        "emb3_eta",
        "emb3_phi",
        "hab1_eta",
        "hab1_phi",
        "hab2_eta",
        "hab2_phi",
        "hab3_eta",
        "hab3_phi",
    ],
    "eta_granularity": {
        "psb": 0.0251791,
        "emb1": 0.0031264,
        "emb2": 0.0251791,
        "emb3": 0.050358,
        "hab1": 0.1007168,
        "hab2": 0.1007168,
        "hab3": 0.0503584,
    },
    "phi_granularity": {
        "psb": 0.098174,
        "emb1": 0.098174,
        "emb2": 0.0245437,
        "emb3": 0.0245437,
        "hab1": 0.098174,
        "hab2": 0.098174,
        "hab3": 0.098174,
    },
}

config_two_layers_Zee = {
    "layers": ["emb1", "emb2"],
    "noise_thresholds": {"emb1": 50, "emb2": 100},
    "dimensions": {
        "emb1": (3, 17),
        "emb2": (7, 9),
    },
    "columns": [
        "event_no",
        "candidate_no",
        "z",
        "et",
        "eta",
        "phi",
        "emb1_eta",
        "emb1_phi",
        "emb2_eta",
        "emb2_phi",
    ],
    "eta_granularity": {
        "emb1": 0.0031264,
        "emb2": 0.0251791,
    },
    "phi_granularity": {
        "emb1": 0.098174,
        "emb2": 0.0245437,
    },
}


class Config:
    def __init__(self, config):
        self.layers = config.get("layers", [])
        self.noise_thresholds = config["noise_thresholds"]
        self.dimensions = config["dimensions"]
        self.columns = config.get("columns", [])
        self.eta_granularity = config["eta_granularity"]
        self.phi_granularity = config["phi_granularity"]

        column_ranges = {}
        for layer in config["layers"]:
            start, end = config["dimensions"][layer]
            column_ranges[f"{layer}_cells"] = (
                f"{layer}_(0,0)",
                f"{layer}_({start - 1},{end - 1})",
            )

        self.column_ranges = column_ranges  # for LDF


FIVE_LAYERS = Config(config_five_layers)
JET_CANDIDATE = Config(config_jet_candidate)
TWO_LAYERS_ZEE = Config(config_two_layers_Zee)

configurations = {
    "five_layers": FIVE_LAYERS,
    "jet_candidate": JET_CANDIDATE,
    "two_layers_Zee": TWO_LAYERS_ZEE,
}
