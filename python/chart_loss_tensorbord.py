import tensorboardX as tbx

#出力先のファイル名指定
writer = tbx.SummaryWriter("../logs/exp-1")

with open("../logs/loss.txt") as f:
    line = f.readline().strip()
    for i in line:
        num = line.split(',')
        writer.add_scalar("group/epoch", num[1], num[0])

writer.close()