# -- coding: utf-8 --
if __name__ == '__main__':
    import os
    import train

    # os.environ["CUDA_VISIBLE_DEVICES"] = "5" # 指定序号的GPU训练

    print("Start")
    train.train()
    print("End")


