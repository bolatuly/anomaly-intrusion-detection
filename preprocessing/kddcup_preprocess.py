import pandas as pd

if __name__ == '__main__':
    headers = pd.read_table("../data/names.txt", header=None).values.ravel()
    data = pd.read_csv("../data/kddcup.data_10_percent_corrected", header=None, names=headers)

    data = pd.get_dummies(data, columns=["protocol_type", "service", "flag"])

    # transform values
    mapping_type = {
        'normal.': 'normal',

        'back.': 'DoS',
        'land.': 'DoS',
        'neptune.': 'DoS',
        'pod.': 'DoS',
        'smurf.': 'DoS',
        'teardrop.': 'DoS',
        'mailbomb.': 'DoS',
        'apache2.': 'DoS',
        'processtable.': 'DoS',
        'udpstorm.': 'DoS',

        'ipsweep.': 'Probe',
        'nmap.': 'Probe',
        'portsweep.': 'Probe',
        'satan.': 'Probe',
        'mscan.': 'Probe',
        'saint.': 'Probe',

        'ftp_write.': 'R2L',
        'guess_passwd.': 'R2L',
        'imap.': 'R2L',
        'multihop.': 'R2L',
        'phf.': 'R2L',
        'spy.': 'R2L',
        'warezclient.': 'R2L',
        'warezmaster.': 'R2L',
        'sendmail.': 'R2L',
        'named.': 'R2L',
        'snmpgetattack.': 'R2L',
        'snmpguess.': 'R2L',
        'xlock.': 'R2L',
        'xsnoop.': 'R2L',
        'worm.': 'R2L',

        'buffer_overflow.': 'U2R',
        'loadmodule.': 'U2R',
        'perl.': 'U2R',
        'rootkit.': 'U2R',
        'httptunnel.': 'U2R',
        'ps.': 'U2R',
        'sqlattack.': 'U2R',
        'xterm.': 'U2R'
    }
    data['label'].replace(mapping_type, inplace=True)

    data["label"] = data["label"].astype('category')
    data["label_cat"] = data["label"].cat.codes

    data.to_csv("../data/csv/kddcup.data_10_percent_corrected.csv", date_format='%Y-%m-%d %H:%M:%S')