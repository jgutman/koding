nohup python ../koding/grid_search_d2v.py -context 5 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con5_dimall_epoch5.out
(started 11:26 am)

nohup python ../koding/grid_search_d2v.py -context 6 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con6_dimall_epoch5.out

nohup python ../koding/grid_search_d2v.py -context 8 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con8_dimall_epoch5.out
(started 11:27 am)

nohup python ../koding/grid_search_d2v.py -context 10 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con10_dimall_epoch5.out

nohup python ../koding/grid_search_d2v.py -context 12 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con12_dimall_epoch5.out

nohup python ../koding/grid_search_d2v.py -context 15 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con15_dimall_epoch5.out

nohup python ../koding/grid_search_w2v.py -context 5 6 -dims 100 150 200 250 300 -epochs 5 -cores 8 > w2vtune/logs/con5_6_dim100_300_epoch5.out
(started 11:26 am)

nohup python ../koding/grid_search_w2v.py -context 8 10 -dims 100 150 200 250 300 -epochs 5 -cores 8 > w2vtune/logs/con8_10_dim100_300_epoch5.out
(similar job running)

nohup python ../koding/grid_search_w2v.py -context 5 6 -dims 100 150 200 250 300 -epochs 5 -cores 8 -unweighted > w2vtune/logs/con5_6_dim100_300_epoch5.out

nohup python ../koding/grid_search_w2v.py -context 8 10 -dims 100 150 200 250 300 -epochs 5 -cores 8 -unweighted > w2vtune/logs/con8_10_dim100_300_epoch5.out