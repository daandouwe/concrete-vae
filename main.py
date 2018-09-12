#!/usr/bin/env python
import argparse

import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train'],
                        help='training or something else.')
    parser.add_argument('--data', type=str, default='/Users/daan/data/mnist/processed',
                        help='where is you mnist?')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--latent-dim', type=int, default=20, metavar='N',
                        help='dimension of latent variable')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--temp-anneal', type=float, default=0.00009)

    parser.add_argument('--temp-interval', type=float, default=300)

    parser.add_argument('--min-temp', type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == 'train':
        train.main(args)


if __name__ == '__main__':
    main()
