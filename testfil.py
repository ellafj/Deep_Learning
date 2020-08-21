def read_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    i = 0

    for line in lines:
        if i != 0 and i != 1:
            line = line.split()
            line.pop(0)
            print(line)
        i += 1

read_file('test.txt')
