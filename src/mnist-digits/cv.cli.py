#!/usr/bin/python3

DEBUG = False

__all__ = []
__version__ = "0.0.1"
__date__ = '2019-01-07'
__updated__ = '2019-01-07'


def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    from sys import argv as Argv

    if argv is None:
        argv = Argv
        pass
    else:
        Argv.extend(argv)
        pass

    from os.path import basename
    program_name = basename(Argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    try:
        program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    except:
        program_shortdesc = __import__('__main__').__doc__
    program_license = '''%s

  Created by Wojciech Migda on %s.
  Copyright 2019 Wojciech Migda. All rights reserved.

  Licensed under the MIT License

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        from argparse import ArgumentParser
        from argparse import RawDescriptionHelpFormatter
        from argparse import FileType
        from sys import stdout, stdin

        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)


        parser.add_argument("-N", "--neval",
            type=int, default=30, action='store', dest="neval",
            help="Number of hyper-parameter sets evaluations")

        parser.add_argument("-F", "--nfolds",
            type=int, default=5, action='store', dest="nfolds",
            help="Number of cross-validation folds")

        parser.add_argument("--cv-jobs",
            type=int, default=1, action='store', dest="ncvjobs",
            help="Number of cross-validation (cross_val_score) jobs")

        parser.add_argument("-j", "--jobs",
            type=int, default=1, action='store', dest="njobs",
            help="Model: number of jobs")

        parser.add_argument("-s", "--seed",
            type=int, default=1, action='store', dest="seed",
            help="Model: RNG seed value")

        parser.add_argument("--number_of_pos_neg_clauses_per_label",
            type=int, default=5, action='store', dest="number_of_pos_neg_clauses_per_label",
            help="Model: number of either positive or negative clauses per label")

        parser.add_argument("--nepochs",
            type=int, default=30, action='store', dest="nepochs",
            help="Model: number of epochs")

        parser.add_argument("--states-range",
            type=str, default="500,2000,20", action='store', dest="states_range",
            help="Model: number of states, 'min,max,step'")

        parser.add_argument("--threshold-range",
            type=str, default="5,20,1", action='store', dest="threshold_range",
            help="Model: threshold, 'min,max,step'")

        parser.add_argument("--s-range",
            type=str, default="1.0,6.0", action='store', dest="s_range",
            help="Model: s, 'min,max'")

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass

        from cv_core import work
        work(
            args.neval,
            args.nfolds,
            args.ncvjobs,
            args.njobs,
            args.seed,
            args.number_of_pos_neg_clauses_per_label,
            args.nepochs,
            args.states_range,
            args.threshold_range,
            args.s_range
            )

        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG:
            raise(e)
            pass
        indent = len(program_name) * " "
        from sys import stderr
        stderr.write(program_name + ": " + repr(e) + "\n")
        stderr.write(indent + "  for help use --help")
        return 2

    pass


if __name__ == "__main__":
    if DEBUG:
        from sys import argv
        argv.append("-h")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
