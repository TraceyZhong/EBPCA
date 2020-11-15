import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prior", type=str, help="enter univariate prior",
                    default='Uniform', const='Uniform', nargs='?')
args = parser.parse_args()
prior = args.prior

prior_prefix = 'univariate/' + prior

if not os.path.exists('figures/' + prior_prefix):
   os.makedirs('figures/' + prior_prefix)

# fig_prefix=fig_prefix + 's_%.1f_gamma_%.1f' % (s_star, gamma)