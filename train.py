import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    print(flags.num_actors)
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
