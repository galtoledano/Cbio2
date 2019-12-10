import argparse
import numpy as np
import math
from scipy.special import logsumexp

converting_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "^": 4, "$": 5}
LINE_LENGTH = 50


# def letters_to_numbers(seq, d):
#     """
#     converting by dictionary dna to numbers and  numbers to dna
#     :param seq: the first alignment
#     :param d: the dictionary
#     :return: the converted seq
#     """
#     keys, choices = list(zip(*d.items()))
#     seq_a = np.array(keys)[:, None, None] == seq
#     seq = np.select(seq_a, choices)[0]
#     return seq


def init_tau(k, p, q):  # whereas k = k + 4
    tau_matrix = np.zeros((k, k))
    for row in range(2, k - 2):
        tau_matrix[row][row + 1] = 1
    tau_matrix[0][1] = q
    tau_matrix[0][k - 2] = 1 - q
    tau_matrix[1][1] = 1 - p
    tau_matrix[1][2] = p
    tau_matrix[k - 2][k - 2] = 1 - p
    tau_matrix[k - 2][k - 1] = p
    with np.errstate(divide='ignore'):
        tau_matrix = np.log(tau_matrix)
    return tau_matrix


def edit_sequence(seq):
    return "^" + seq + "$"


def read_emission(file):
    mat = np.genfromtxt(file, delimiter='\t', skip_header=1)
    B_1_2 = np.full(4, 0.25)
    mat = np.vstack([mat, B_1_2])
    B_end_B_start = np.zeros(4)
    mat = np.vstack([mat, B_end_B_start])
    mat = np.vstack([B_1_2, mat])
    mat = np.vstack([B_end_B_start, mat])
    k = mat.shape[0]
    hat_vector = np.zeros((1, k))
    dollar_vector = np.zeros((1, k))
    hat_vector[0][0], dollar_vector[0][k - 1] = 1, 1
    mat = np.insert(mat, 4, hat_vector, axis=1)
    mat = np.insert(mat, 5, dollar_vector, axis=1)
    with np.errstate(divide='ignore'):
        mat = np.log(mat)
    return mat, k


def viterbi(seq, tau, emission_table):
    k = emission_table.shape[0]
    v_matrix = np.zeros((k, len(seq)))
    t_matrix = np.zeros((k, len(seq)))
    v_matrix[0][0], t_matrix[0][0] = math.log(1), math.log(1)
    for letter in range(1, len(seq)):
        # for state in range(k):  #todo
        # prev_col = v_matrix[:, letter-1]
        # tau_col = tau[:, state]
        # temp_mult = prev_col + tau_col
        sum_of_cols = v_matrix[:, letter-1].reshape(-1, 1) + tau
        max_val = np.max(sum_of_cols,  axis=0).T
        argmax_index = np.argmax(sum_of_cols, axis=0).T
        v_matrix[:, letter] = max_val + emission_table[:, converting_dict[seq[letter]]]
        t_matrix[:, letter] = argmax_index
    return v_matrix, t_matrix


def trace_viterbi(v_matrix, t_matrix):
    viterbi_seq = ""
    last_col = v_matrix[:, len(v_matrix[0]) - 1]
    curr = np.where(last_col == max(last_col))[0][0]
    length = len(v_matrix) - 1
    for letter in range(len(v_matrix[0]) - 1, 1, -1):  # until 1? 0?
        curr = t_matrix[int(curr)][letter]
        if curr == 0 or curr == 1 or curr == length or curr == length - 1:
            viterbi_seq = "B" + viterbi_seq
        else:
            viterbi_seq = "M" + viterbi_seq
    return viterbi_seq


def print_viterbi_output(original_seq, viterbi_seq):
    index_org_seq = 0
    index_viterbi_seq = 0
    while index_org_seq != len(original_seq):
        leftover_length = len(original_seq) - index_org_seq
        if leftover_length <= LINE_LENGTH:
            for i in range(leftover_length):
                print(viterbi_seq[index_viterbi_seq], end="")
                index_viterbi_seq += 1
            print()
            for i in range(leftover_length):
                print(original_seq[index_org_seq], end="")
                index_org_seq += 1
        else:
            for i in range(LINE_LENGTH):
                print(viterbi_seq[index_viterbi_seq], end="")
                index_viterbi_seq += 1
            print()
            for i in range(LINE_LENGTH):
                print(original_seq[index_org_seq], end="")
                index_org_seq += 1
            print("\n")


def forward_algorithm(seq, tau_mat,  emission_matrix, k):
    f_mat = np.zeros((k, len(seq)))
    f_mat[0][0] = 1
    with np.errstate(divide='ignore'):
        f_mat = np.log(f_mat)
    for letter in range(1, len(seq)):
        sum_of_cols = f_mat[:, letter-1].reshape(-1, 1) + tau_mat
        f_mat[:, letter] = logsumexp(sum_of_cols, axis=0) + emission_matrix[:, converting_dict[seq[letter]]]
    return f_mat


def backward_algorithm(seq, tau_mat,  emission_matrix, k):
    b_mat = np.zeros((k, len(seq)))
    b_mat[-1][-1] = 1
    with np.errstate(divide='ignore'):
        b_mat = np.log(b_mat)
    for letter in range(len(seq)-1, 0, -1):
        sum_of_cols = b_mat[:, letter].reshape(-1, 1) + tau_mat.T + emission_matrix[:, converting_dict[seq[letter]]].reshape(-1,1)
        b_mat[:, letter-1] = logsumexp(sum_of_cols, axis=0)
    return b_mat


def posterior(f_mat, b_mat, k, seq):
    post_mat = np.zeros((k, len(seq)))
    with np.errstate(divide='ignore'):
        post_mat = np.log(post_mat)
    curr = 0
    post_seq = ""
    post_mat = f_mat + b_mat
    for letter in range(len(post_mat[0]) - 1, 1, -1):
        letter_col = post_mat[:, letter]
        curr = np.where(letter_col == max(letter_col))[0][0]
        # curr = post_mat[int(curr)][letter]
        if curr == 0 or curr == 1 or curr == k - 1 or curr == k - 2:
            post_seq = "B" + post_seq
        else:
            post_seq = "M" + post_seq
    return post_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()

    emission_table, k = read_emission(args.initial_emission)
    p = args.p
    q = args.q
    tau = init_tau(k, p, q)
    seq = edit_sequence(args.seq)

    if args.alg == 'viterbi':
        v_matrix, t_matrix = viterbi(seq, tau, emission_table)
        viterbi_seq = trace_viterbi(v_matrix, t_matrix)
        # return viterbi_seq
        print_viterbi_output(args.seq, viterbi_seq)

    elif args.alg == 'forward':
        f_mat = forward_algorithm(seq, tau,  emission_table, k)
        print(f_mat[-1][-1])

    elif args.alg == 'backward':
        b_mat = backward_algorithm(seq, tau, emission_table, k)
        print(b_mat[0][0])

    elif args.alg == 'posterior':
        f_mat = forward_algorithm(seq, tau,  emission_table, k)
        b_mat = backward_algorithm(seq, tau, emission_table, k)
        post_seq = posterior(f_mat, b_mat, k, seq)
        print_viterbi_output(args.seq, post_seq)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()


