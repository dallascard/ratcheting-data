import os
import copy
import json
from subprocess import call
from optparse import OptionParser

import numpy as np


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='/u/scr/dcard/projects/scientific-productivity/output/matched_data.csv',
                      help='Input file: default=%default')
    parser.add_option('--base-year', type=int, default=2009,
                      help='Year for measuring effect sizes: default=%default')
    parser.add_option('--base-age', type=int, default=0,
                      help='Age for measuring effect sizes: default=%default')
    parser.add_option('--max-comp', type=int, default=20,
                      help='Max components to try: default=%default')
    parser.add_option('--only-faculty', action="store_true", default=False,
                      help='Use only faculty: default=%default')
    parser.add_option('--no-faculty', action="store_true", default=False,
                      help='Exclude faculty: default=%default')


    (options, args) = parser.parse_args()

    infile = options.infile
    base_year = options.base_year
    base_age = options.base_age
    max_comp = options.max_comp
    only_faculty = options.only_faculty
    no_faculty = options.no_faculty

    target = 'papers_all'
    config = {
        "infile": infile,
        "target": target,
        "max_iter": 500,
        "factors": [],
        "interactions": [],
    }

    if only_faculty:
        config['subset_column'] = 'faculty'
        config['subset_target'] = 1
        outdir = os.path.join('..', 'results', 'only_faculty')
    elif no_faculty:
        config['subset_column'] = 'faculty'
        config['subset_target'] = 0
        outdir = os.path.join('..', 'results', 'no_faculty')
    else:
        outdir = os.path.join('..', 'results', 'all')

    orig_factors = {'grad_year': {'name': 'grad_year', 'linear': True, 'type': 'int', 'group': 'cohort', 'pred_val': base_year},
                    'age': {'name': 'age', 'linear': True, 'type': 'int', 'group': 'age', 'pred_val': base_age}}

    orig_interactions = [['grad_year_1', 'age_1']]

    config['fold'] = None
    config['train_file'] = infile
    config['test_file'] = None

    for rnd in range(0, max_comp):

        factors = copy.deepcopy(orig_factors)
        expdir = os.path.join(outdir, target, f'{rnd:03d}')
        makedirs(expdir)
        factors['grad_year']['components_excl_linear'] = rnd
        config['factors'] = [factor for factor in factors.values()]
        interactions = copy.deepcopy(orig_interactions)
        config['interactions'] = interactions
        write_to_json(config, os.path.join(expdir, 'config.json'))
        cmd = ['python', 'run_sm_model.py', os.path.join(expdir, 'config.json')]
        print(' '.join(cmd))
        call(cmd)

    bics = []
    for rnd in range(0, max_comp):
        expdir = os.path.join(outdir, target, f'{rnd:03d}')
        infile = os.path.join(expdir, 'report.json')
        with open(infile) as f:
            report = json.load(f)
        bics.append(report['bic'])

    order = np.argsort(bics)
    n_comp_excl_linear = int(order[0])
    print("Using {:d} components excluding linear".format(n_comp_excl_linear))

    str_to_try = ['thesis_topic']

    rnd = 100
    for factor in str_to_try:
        factors = copy.deepcopy(orig_factors)
        expdir = os.path.join(outdir, target, f'{rnd:03d}')
        makedirs(expdir)
        factors['grad_year']['components_excl_linear'] = n_comp_excl_linear
        factors[factor] = {'name': factor, 'type': 'str', 'transform': None, 'exclude_most_common': True, 'min_count': 0, 'group': 'other'}
        config['factors'] = [factor for factor in factors.values()]
        interactions = copy.deepcopy(orig_interactions)
        config['interactions'] = interactions
        write_to_json(config, os.path.join(expdir, 'config.json'))
        cmd = ['python', 'run_sm_model.py', os.path.join(expdir, 'config.json')]
        print(' '.join(cmd))
        call(cmd)
        rnd += 1

    vec_to_try = ['av_num_authors_per_person',
                  'mean_paper_length_per_person',
                  'venue_diffs',
                  'author_rel_sim',
                  'institutional_prestige'
                  ]

    for factor in vec_to_try:
        factors = copy.deepcopy(orig_factors)
        expdir = os.path.join(outdir, target, f'{rnd:03d}')
        makedirs(expdir)
        factors['grad_year']['components_excl_linear'] = n_comp_excl_linear
        factors[factor] = {'name': factor, 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
        config['factors'] = [factor for factor in factors.values()]
        interactions = copy.deepcopy(orig_interactions)
        config['interactions'] = interactions
        write_to_json(config, os.path.join(expdir, 'config.json'))
        cmd = ['python', 'run_sm_model.py', os.path.join(expdir, 'config.json')]
        print(' '.join(cmd))
        call(cmd)
        rnd += 1

    inter_to_try = ['top_20']

    for factor in inter_to_try:
        factors = copy.deepcopy(orig_factors)
        expdir = os.path.join(outdir, target, f'{rnd:03d}')
        makedirs(expdir)
        factors['grad_year']['components_excl_linear'] = n_comp_excl_linear
        factors[factor] = {'name': factor, 'type': 'vector', 'transform': None, 'group': 'period'}
        config['factors'] = [factor for factor in factors.values()]
        interactions = copy.deepcopy(orig_interactions)
        interactions.append([factor, 'grad_year_1'])
        config['interactions'] = interactions
        write_to_json(config, os.path.join(expdir, 'config.json'))
        cmd = ['python', 'run_sm_model.py', os.path.join(expdir, 'config.json')]
        print(' '.join(cmd))
        call(cmd)
        rnd += 1

    factors = copy.deepcopy(orig_factors)
    expdir = os.path.join(outdir, target, f'{rnd:03d}')
    makedirs(expdir)
    factors['grad_year']['components_excl_linear'] = n_comp_excl_linear
    factors['thesis_topic'] = {'name': 'thesis_topic', 'type': 'str', 'transform': None, 'exclude_most_common': True, 'min_count': 0, 'group': 'other'}
    factors['av_num_authors_per_person'] = {'name': 'av_num_authors_per_person', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['mean_paper_length_per_person'] = {'name': 'mean_paper_length_per_person', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['venue_diffs'] = {'name': 'venue_diffs', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['author_rel_sim'] = {'name': 'author_rel_sim', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['institutional_prestige'] = {'name': 'institutional_prestige', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['top_20'] = {'name': 'top_20', 'type': 'vector', 'transform': None, 'group': 'period'}
    config['factors'] = [factor for factor in factors.values()]
    interactions = copy.deepcopy(orig_interactions)
    interactions.append(['top_20', 'grad_year_1'])
    config['interactions'] = interactions
    write_to_json(config, os.path.join(expdir, 'config.json'))
    cmd = ['python', 'run_sm_model.py', os.path.join(expdir, 'config.json')]
    print(' '.join(cmd))
    call(cmd)
    rnd += 1

    # adding gender
    factors = copy.deepcopy(orig_factors)
    expdir = os.path.join(outdir, target, f'{rnd:03d}')
    makedirs(expdir)
    factors['grad_year']['components_excl_linear'] = n_comp_excl_linear
    factors['thesis_topic'] = {'name': 'thesis_topic', 'type': 'str', 'transform': None, 'exclude_most_common': True, 'min_count': 0, 'group': 'other'}
    factors['av_num_authors_per_person'] = {'name': 'av_num_authors_per_person', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['mean_paper_length_per_person'] = {'name': 'mean_paper_length_per_person', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['venue_diffs'] = {'name': 'venue_diffs', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['author_rel_sim'] = {'name': 'author_rel_sim', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['institutional_prestige'] = {'name': 'institutional_prestige', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['topic_prop'] = {'name': 'topic_prop', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['cs_funding_period_prev_3'] = {'name': 'cs_funding_period_prev_3', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['phd_grads_period_next_3'] = {'name': 'phd_grads_period_next_3', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['nces_bs_period_prev_3'] = {'name': 'nces_bs_period_prev_3', 'type': 'vector', 'transform': 'zscore', 'group': 'period'}
    factors['top_20'] = {'name': 'top_20', 'type': 'vector', 'transform': None, 'group': 'period'}
    factors['typically_female_name'] = {'name': 'typically_female_name', 'type': 'vector', 'transform': None, 'group': 'period'}
    config['factors'] = [factor for factor in factors.values()]
    interactions = copy.deepcopy(orig_interactions)
    interactions.append(['top_20', 'grad_year_1'])
    interactions.append(['typically_female_name', 'grad_year_1'])
    config['interactions'] = interactions
    write_to_json(config, os.path.join(expdir, 'config.json'))
    cmd = ['python', 'run_sm_model.py', os.path.join(expdir, 'config.json')]
    print(' '.join(cmd))
    call(cmd)
    rnd += 1


def makedirs(name):
    if not os.path.exists(name):
        os.makedirs(name)


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with open(output_filename, 'w') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)



if __name__ == '__main__':
    main()
