import numpy as np
import pandas as pd

class DataExtractor:
    def __init__(self, dataset_path, train_file, test_file):
        self.dataset_path = dataset_path
        self.train_file = train_file
        self.test_file = test_file
        self.attacks_subClass = [
            ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'], 
            ['ipsweep', 'mscan', 'portsweep', 'saint', 'satan', 'nmap'],
            ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'],
            ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
        ]

    def create_dataset(self, scenario, train_attacks, test_attacks):
        # Extract train and test datasets
        train_dataset = self._extract_subset(self.train_file, train_attacks)
        test_dataset = self._extract_subset(self.test_file, test_attacks)

        # Save datasets with scenario-specific filenames
        train_filename = self.dataset_path + f"custom_train_{scenario}.csv"
        test_filename = self.dataset_path + f"custom_test_{scenario}.csv"
        train_dataset.to_csv(train_filename, index=False)
        test_dataset.to_csv(test_filename, index=False)
        return train_filename, test_filename

    def _extract_subset(self, file, attack_classes):
        dataset = pd.read_csv(self.dataset_path + file, header=None)
        filtered_data = dataset[dataset.iloc[:, -2].apply(lambda x: self._filter_attack_class(x, attack_classes))]
        return filtered_data

    def _filter_attack_class(self, label, attack_classes):
        for attack_class in attack_classes:
            if label in self.attacks_subClass[attack_class - 1]:
                return True
        return label == 'normal'

# Initialize DataExtractor
dataset_path = ''
train_file = 'KDDTrain+.txt'
test_file = 'KDDTest+.txt'
data_extractor = DataExtractor(dataset_path, train_file, test_file)

# Define scenarios
scenarios = {
    'SA': {'train': [1, 3], 'test': [2, 4]},
    'SB': {'train': [1, 2], 'test': [1]},
    'SC': {'train': [1, 2], 'test': [1, 2, 3]}
}

# Extract and save datasets for each scenario
generated_files = {}
for scenario, attack_classes in scenarios.items():
    train_filename, test_filename = data_extractor.create_dataset(scenario, attack_classes['train'], attack_classes['test'])
    generated_files[scenario] = {'train': train_filename, 'test': test_filename}

generated_files
