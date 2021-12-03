import numpy as np
import itertools
args = None

OpportunityOpt = {
    'batch_size': 32,
    'seq_len': 60,
    'input_dim': 6,
    'learning_rate': 0.0001,
    'weight_decay': 0.00001,
    'file_path': '/mnt/sting/adiorz/mobile_sensing/datasets/opportunity_std_scaled_all.csv',
    # 4 x 7 = 28 available domains
    'users': ['s' + str(i) for i in range(1, 5)],  # all available users
    # 'positions': ['RUA', 'LUA', 'RLA', 'LLA', 'Back', 'L_Shoe', 'R_Shoe'],  # complete IMU sensor data only
    'positions': ['RUA', 'LUA', 'RLA', 'LLA'],  # complete IMU sensor data only

    'classes': ['stand', 'walk', 'sit', 'lie'],  # remove null class
    'num_class': 4,
    'num_trans': 11,  # number of transformations
}

# processed_domains = list(np.random.permutation(args.opt['domains'], size=args.opt['num_src']))
def init_domains():

    src_users = list(np.random.permutation(OpportunityOpt['users'])[:2])
    src_poss = list(np.random.permutation(OpportunityOpt['positions'])[:5])
    tgt_users = list(set(OpportunityOpt['users']) - set(src_users))
    tgt_poss = list(set(OpportunityOpt['positions']) - set(src_poss))
    OpportunityOpt['src_domains'] = sorted(list(itertools.product(src_users, src_poss)))
    OpportunityOpt['tgt_domains'] = sorted(list(itertools.product(tgt_users, tgt_poss)))

if __name__=="__main__":
    init_domains()