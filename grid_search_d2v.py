import build_tune_d2v
import sys, argparse, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-context", dest = "context", type = int, nargs = "+")
    parser.add_argument("-dims", dest = "dims", type = int, nargs = "+")
    parser.add_argument("-cores", dest = "cores", type = int)
    parser.add_argument("-epochs", dest = "epochs", type = int)
    parser.set_defaults(cores = 4, epochs = 1, context = [5], dims = [100])
    args = parser.parse_args()
    
    for context in args.context:
        for dims in args.dims:
            experiment = build_tune_d2v.argdict(context, dims)
            experiment.cores = args.cores
            experiment.epochs = args.epochs
            experiment.root_dir = "/scratch/jg3862/gdrive"
            build_tune_d2v.main(args = experiment) 

if __name__ == '__main__':
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main()
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()