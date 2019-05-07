from subprocess import call


print("Creating classified dataset")
for i in range(1, 4):
    call(['python3', 'data_treatment.py', '-c', 'MX',
          '-k', str(i), '-o', f'../data/mexico_k_{i}.csv'])

    for l in [1, 3, 5, 7]:
        call(['python3', 'data_treatment.py', '-i', '../data/mexico_k_{}.csv'.format(i),
              '-o', '../data/test/mexico_k_{}_layers_{}.csv'.format(i, l), '-m', str(l)])

    for d in [10, 30, 60]:
        call(['python3', 'data_treatment.py', '-i', '../data/mexico_k_{}.csv'.format(i),
              '-o', '../data/test/mexico_k_{}_depth_{}.csv'.format(i, d), '-d', str(d)])

    for d in [10, 30, 60]:
        call(['python3', '03_merge_depth.py', '-i', '../data/mexico_k_{}.csv'.format(i),
              '-o', '../data/test/mexico_k_{}_depth_not_weighted_{}.csv'.format(i, d), '-d', str(d)])

    call(['python3', '03_merge_standard.py', '-i', '../data/mexico_k_{}.csv'.format(i),
          '-o', '../data/test/mexico_k_{}_standard.csv'.format(i)])
