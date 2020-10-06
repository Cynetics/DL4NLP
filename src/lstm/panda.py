import pandas as pd 

df = pd.read_csv('classification_report_DROPOUT_BOY.csv')
lowest_prec = df.sort_values('precision')[['lan', 'precision' ]]
lowest_rec = df.sort_values('recall')[['lan', 'recall']]
lowest_f1 = df.sort_values('f1-score')[['lan', 'f1-score' ]]

with pd.option_context('display.max_rows', None): 
    print(lowest_prec)
    print("recall")
    print(lowest_rec)
    print("f1")
    print(lowest_f1)