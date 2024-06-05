from models.segformer import Jigsaw_Solver

def get_model(name):
    if name == "segformer":
        return Jigsaw_Solver()
    raise  ValueError(f"No module name '{name}'")
