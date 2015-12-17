import build_tune_d2v
import sys, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-context", dest = "context", type = int, nargs = "+")
    parser.add_argument("-dims", dest = "dims", type = int, nargs = "+")
    parser.set_defaults()
    args = parser.parse_args()
    
    print type(args), len(args), args

if __name__ == '__main__':
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main()
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()