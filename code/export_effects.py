import os
import json
from optparse import OptionParser
from collections import defaultdict

import numpy as np
import pandas as pd


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    basedirs = [os.path.join('..', 'results', 'no_faculty', 'papers_all', '108'),
                os.path.join('..', 'results', 'only_faculty', 'papers_all', '108')]

    effects_both = defaultdict(list)
    lower_both = defaultdict(list)
    upper_both = defaultdict(list)

    for b_i, basedir in enumerate(basedirs):
        mean_df = pd.read_csv(os.path.join(basedir, 'params.csv'), header=0, index_col=0)
        gradyear_matrix = np.load(os.path.join(basedir, 'grad_year.npz'))['matrix']
        age_matrix = np.load(os.path.join(basedir, 'age.npz'))['matrix']
        intervals_df = pd.read_csv(os.path.join(basedir, 'intervals.csv'), header=0, index_col=0)
        with open(os.path.join(basedir, 'zscore_stds.json')) as f:
            zscore_stds = json.load(f)

        factors = ['av_num_authors_per_person', 'top_20', 'typically_female_name', 'institutional_prestige', 'mean_paper_length_per_person',  'author_rel_sim', 'venue_diffs', 'topic_prop', 'cs_funding_period_prev_3', 'phd_grads_period_next_3', 'nces_bs_period_prev_3']

        rows = [r for r in mean_df.index if r.startswith('grad_year_') and not r.startswith('grad_year_1_X')]
        means = np.array([mean_df.loc[r, '0'] for r in rows])
        for year in [-1]:
            grad_year_transform = gradyear_matrix[year, :]
            grad_year_2009_mean = np.dot(means, grad_year_transform)

            rows = [r for r in mean_df.index if r.startswith('age_') and not r.startswith('age_1_X')]
            means = np.array([mean_df.loc[r, '0'] for r in rows])
            for age in [4]:
                age_transform = age_matrix[age, :]
                age_0_mean = np.dot(means, age_transform)

                const = mean_df.loc['const', '0']

                interaction_name = 'grad_year_1_X_age_1'
                interaction_val = mean_df.loc[interaction_name, '0']

                gender_interaction_name = 'typically_female_name_X_grad_year_1'
                gender_interaction_val = mean_df.loc[gender_interaction_name, '0']

                top20_interaction_name = 'top_20_X_grad_year_1'
                top20_interaction_val = mean_df.loc[top20_interaction_name, '0']

                for factor in factors:
                    if factor == 'top_20' or factor == 'typically_female_name':
                        factor_val = mean_df.loc[factor, '0']
                        lower = intervals_df.loc[factor, '0']
                        upper = intervals_df.loc[factor, '1']
                        divisor = 1.
                    else:
                        name = 'zscore(' + factor + ')'
                        factor_val = mean_df.loc[name, '0']
                        lower = intervals_df.loc[name, '0']
                        upper = intervals_df.loc[name, '1']
                        if factor in {'author_rel_sim','institutional_prestige', 'topic_prop'}:
                            divisor = 1.
                        elif factor == 'cs_funding_period_prev_3':
                            divisor = zscore_stds[factor] / 3000000.
                        elif factor == 'phd_grads_period_next_3':
                            divisor = zscore_stds[factor] / 300.
                        elif factor == 'nces_bs_period_prev_3':
                            divisor = zscore_stds[factor] / 30000.
                        else:
                            divisor = zscore_stds[factor]

                    baseline = const + grad_year_2009_mean + age_0_mean + grad_year_transform[0] * age_transform[0] * interaction_val
                    total_effect = const + grad_year_2009_mean + age_0_mean + grad_year_transform[0] * age_transform[0] * interaction_val + factor_val / divisor
                    lower_effect = const + grad_year_2009_mean + age_0_mean + grad_year_transform[0] * age_transform[0] * interaction_val + lower / divisor
                    upper_effect = const + grad_year_2009_mean + age_0_mean + grad_year_transform[0] * age_transform[0] * interaction_val + upper / divisor

                    if factor == 'typically_female_name':
                        total_effect += gender_interaction_val * grad_year_transform[0]
                        lower_effect += gender_interaction_val * grad_year_transform[0]
                        upper_effect += gender_interaction_val * grad_year_transform[0]

                    elif factor == 'top_20':
                        total_effect += top20_interaction_val * grad_year_transform[0]
                        lower_effect += top20_interaction_val * grad_year_transform[0]
                        upper_effect += top20_interaction_val * grad_year_transform[0]

                    effect = np.exp(total_effect) - np.exp(baseline)
                    lower = np.exp(lower_effect) - np.exp(baseline)
                    upper = np.exp(upper_effect) - np.exp(baseline)
                    effects_both[b_i].append(effect)
                    lower_both[b_i].append(lower)
                    upper_both[b_i].append(upper)

    names = ['+1 co-author on average',
             'Top 20 school',
             'Typically-female name',
             'Inst. prestige (+1 s.d.)',
             'Paper length (+1 page)',
             'Self-similarity (+1 s.d.)',
             'Venue adjustment (+1\%)',
             'Field popularity (+1 s.d.)',
             'NSF funding (+\$1m/year)',
             'Future PhDs (+100/year)',
             'Recent BSs (+10k/year)']

    for n_i, name in enumerate(names):
        effect0 = effects_both[0][n_i]
        effect1 = effects_both[1][n_i]
        lower0 = lower_both[0][n_i]
        lower1 = lower_both[1][n_i]
        upper0 = upper_both[0][n_i]
        upper1 = upper_both[1][n_i]
        if lower0 > 0 or upper0 < 0:
            if lower1 > 0 or upper1 < 0:
                print(name + ' & \\textbf{' +  '{:.2f}'.format(effect0) + '} ' + '({:.3f},{:.3f})'.format(lower0, upper0) + ' & \\textbf{' +  '{:.2f}'.format(effect1) + '} ' + '({:.3f},{:.3f})'.format(lower1, upper1) + ' \\\\')
            else:
                print(name + ' & \\textbf{' +  '{:.2f}'.format(effect0) + '} ' + '({:.3f},{:.3f})'.format(lower0, upper0) + ' & ' + '{:.2f}'.format(effect1) + ' ({:.3f},{:.3f})'.format(lower1, upper1) + ' \\\\')
        else:
            if lower1 > 0 or upper1 < 0:
                print(name + ' & ' + '{:.2f}'.format(effect0) + ' ({:.3f},{:.3f})'.format(lower0, upper0) + ' & \\textbf{' +  '{:.2f}'.format(effect1) + '} ' + '({:.3f},{:.3f})'.format(lower1, upper1) + ' \\\\')
            else:
                print(name + ' & ' + '{:.2f}'.format(effect0) +  '({:.3f},{:.3f})'.format(lower0, upper0) + ' & ' + '{:.2f}'.format(effect1) + ' ({:.3f},{:.3f})'.format(lower1, upper1) + ' \\\\')


if __name__ == '__main__':
    main()
