# this is mycode
import sys
import getopt
import argparse

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
      print ('test.py -i <inputfile>')
      sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg
    print ('Input file is "', inputfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--foo")
    parser.add_argument('-i --inputfile', dest='ifile')
    args = parser.parse_args(['-i', 'myfile'])
    print(args.ifile)
    sys.argv = '-i myfile'.split()
    main(sys.argv)