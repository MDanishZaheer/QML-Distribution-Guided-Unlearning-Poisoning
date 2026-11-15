"""Distribution-guided and uniform-target class unlearning."""

from src.unlearning.methods import unlearn
from src.unlearning.targets import make_forget_target, prepare_references

__all__ = ["unlearn", "make_forget_target", "prepare_references"]
