import pandas as pd
import copy

if __name__ == '__main__':
    headers = pd.read_table("../data/kdd_plus/names+.txt", header=None).values.ravel()
    train = pd.read_csv("../data/kdd_plus/KDDTrain+.txt", header=None, names=headers)

    test = pd.read_csv("../data/kdd_plus/KDDTest+.txt", header=None, names=headers)
    #test = pd.read_csv("../data/kdd_plus/KDDTest-21.txt", header=None, names=headers)

    train_objs_num = len(train)
    dataset = pd.concat(objs=[train, test], axis=0)
    dataset = pd.get_dummies(dataset, columns=["protocol_type", "service", "flag"])

    # transform values
    mapping_type = {
        'normal': 'normal',

        'back': 'DoS',
        'land': 'DoS',
        'neptune': 'DoS',
        'pod': 'DoS',
        'smurf': 'DoS',
        'teardrop': 'DoS',
        'mailbomb': 'DoS',
        'apache2': 'DoS',
        'processtable': 'DoS',
        'udpstorm': 'DoS',

        'ipsweep': 'Probe',
        'nmap': 'Probe',
        'portsweep': 'Probe',
        'satan': 'Probe',
        'mscan': 'Probe',
        'saint': 'Probe',

        'ftp_write': 'R2L',
        'guess_passwd': 'R2L',
        'imap': 'R2L',
        'multihop': 'R2L',
        'phf': 'R2L',
        'spy': 'R2L',
        'warezclient': 'R2L',
        'warezmaster': 'R2L',
        'sendmail': 'R2L',
        'named': 'R2L',
        'snmpgetattack': 'R2L',
        'snmpguess': 'R2L',
        'xlock': 'R2L',
        'xsnoop': 'R2L',
        'worm': 'R2L',

        'buffer_overflow': 'U2R',
        'loadmodule': 'U2R',
        'perl': 'U2R',
        'rootkit': 'U2R',
        'httptunnel': 'U2R',
        'ps': 'U2R',
        'sqlattack': 'U2R',
        'xterm': 'U2R'
    }
    dataset['label'].replace(mapping_type, inplace=True)

    dataset["label"] = dataset["label"].astype('category')
    dataset["label_cat"] = dataset["label"].cat.codes

    dataset = dataset.drop(["type"], axis=1)

    train = copy.copy(dataset[:train_objs_num])
    test = copy.copy(dataset[train_objs_num:])

    train.to_csv("../data/csv/KDDTrain+.csv", date_format='%Y-%m-%d %H:%M:%S')
    test.to_csv("../data/csv/KDDTest+.csv", date_format='%Y-%m-%d %H:%M:%S')
    #test.to_csv("../data/csv/KDDTest-21.csv", date_format='%Y-%m-%d %H:%M:%S')

