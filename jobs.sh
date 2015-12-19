nohup python ../koding/grid_search_d2v.py -context 5 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con5_dimall_epoch5.out
(started 11:26 am) (completed)

nohup python ../koding/grid_search_d2v.py -context 6 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con6_dimall_epoch5.out
(started 10:05 pm)

nohup python ../koding/grid_search_d2v.py -context 8 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con8_dimall_epoch5.out
(started 11:27 am) (completed)

nohup python ../koding/grid_search_d2v.py -context 10 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con10_dimall_epoch5.out
(started 10:01 pm on HPC) (job hpc_d2v_exp1.sh)

nohup python ../koding/grid_search_d2v.py -context 12 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con12_dimall_epoch5.out
(job hpc_d2v_exp2.sh)

nohup python ../koding/grid_search_d2v.py -context 15 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con15_dimall_epoch5.out
(job hpc_d2v_exp3.sh)

nohup python ../koding/grid_search_w2v.py -context 5 6 -dims 100 150 200 250 300 -epochs 5 -cores 8 > w2vtune/logs/con5_6_dim100_300_epoch5.out
(started 11:26 am)

nohup python ../koding/grid_search_w2v.py -context 8 10 -dims 100 150 200 250 300 -epochs 5 -cores 8 > w2vtune/logs/con8_10_dim100_300_epoch5.out
(similar job running) (job hpc_d2v_exp6.sh)

nohup python ../koding/grid_search_w2v.py -context 5 6 -dims 100 150 200 250 300 -epochs 5 -cores 8 -unweighted > w2vtune/logs/con5_6_dim100_300_epoch5.out
(job hpc_d2v_exp4.sh)

nohup python ../koding/grid_search_w2v.py -context 8 10 -dims 100 150 200 250 300 -epochs 5 -cores 8 -unweighted > w2vtune/logs/con8_10_dim100_300_epoch5.out
(job hpc_d2v_exp5.sh)