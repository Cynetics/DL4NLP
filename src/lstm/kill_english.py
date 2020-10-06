import numpy as np 


train_half_1 = 'data/wili-2018/x_train_sub_half_1.txt'
train_half_2 = 'data/wili-2018/x_train_sub_half_2.txt'

train_half_1_y = 'data/wili-2018/y_train_sub_half_1.txt'
train_half_2_y = 'data/wili-2018/y_train_sub_half_2.txt'

test = 'data/wili-2018/x_test.txt'
test_y = 'data/wili-2018/y_test.txt'

indices_half1 = 'indices_fucked_half_1.txt'
indices_half2 = 'indices_fucked_half_2.txt'
indices_test  = 'indices_fucked_test.txt'

outfile = 'data/wili-2018/x_train_cleaned.txt'
outfile_y = 'data/wili-2018/y_train_cleaned.txt'

outfile2 = 'data/wili-2018/x_test_cleaned.txt'
outfile_y2 = 'data/wili-2018/y_test_cleaned.txt'
with open(outfile2, 'w') as f:
    b = 9 
with open(outfile_y2, 'w') as f:
    b = 9 

with open(outfile, 'w') as f:
    b = 9 
with open(outfile_y, 'w') as f:
    b = 9 
def get_fucked(file_x, file_y, indices_file, out1, out2):
    

    indices = []
    with open(indices_file, 'r') as f: 
        for z in f.readlines():
            indices.append(int(float(z.strip())))
    
    with open(file_x, 'r') as f:
        for i,p in enumerate(f.readlines()):
            if i not in indices:
                with open(out1, 'a') as f :
                    f.write(p)

    with open(file_y, 'r') as f: 
        for i,p in enumerate(f.readlines()):
            if i not in indices:
                with open(out2, 'a') as f :
                    f.write(p)



get_fucked(train_half_1, train_half_1_y, indices_half1, outfile, outfile_y)
get_fucked(train_half_2, train_half_2_y, indices_half2, outfile, outfile_y)

get_fucked(test, test_y, indices_test, outfile2, outfile_y2)
