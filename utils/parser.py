import argparse

class Parser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Distillation parser")
        self.parser.add_argument("--alpha", type=float, action="store")
        self.parser.add_argument("--n_exp", type=int, action="store")
        self.parser.add_argument("--data_dir", type=str, action="store")
        self.parser.add_argument("--teacher_file", type=str, action="store")
        self.parser.add_argument("--num_epochs", type=int, action="store")
        self.parser.add_argument("--batch_size", type=int, action="store")
        self.parser.add_argument("--lr", type=float, action="store")
        self.parser.add_argument("--momentum", type=float, action="store")
        self.parser.add_argument("--s_lr_step_size", type=float, action="store")
        self.parser.add_argument("--s_rl_gamma", type=float, action="store")


    def parse_args(self):
        args = self.parser.parse_args()
        return args