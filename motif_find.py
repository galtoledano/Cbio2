import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()
    
    if args.alg == 'viterbi':
        raise NotImplementedError

    elif args.alg == 'forward':
        raise NotImplementedError

    elif args.alg == 'backward':
        raise NotImplementedError
    
    elif args.alg == 'posterior':
        raise NotImplementedError


if __name__ == '__main__':
    main()
