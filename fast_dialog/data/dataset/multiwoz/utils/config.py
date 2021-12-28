import os

class Config:
    def __init__(self, version):
        self.version = version
        if self.version == '2.0':
            self.data_prefix = './../MultiWOZ_2.0'
            self.data_path = os.path.join(self.data_prefix, 'annotated_user_da_with_span_full.json')
        elif self.version == '2.1':
            self.data_prefix = './../MultiWOZ_2.1'
            self.data_path = os.path.join(self.data_prefix, 'data.json')
        elif self.version == '2.2':
            self.data_prefix = './../MultiWOZ_2.2'
            self.data_path = os.path.join(self.data_prefix, 'data.json')
        else:
            raise NotImplementedError('没这个版本的数据集')

        self.vocab_path_train = os.path.join(self.data_prefix, 'multi-woz-processed/vocab')
        self.processed_data_path = os.path.join(self.data_prefix, 'multi-woz-processed/')
        self.dev_list = os.path.join(self.data_prefix, 'valListFile.json')
        self.test_list = os.path.join(self.data_prefix, 'testListFile.json')
        self.ontology_path = os.path.join(self.data_prefix, 'ontology.json')
        self.value_set_path = os.path.join(self.data_prefix, 'db', 'value_set.json')
        self.data_for_damd = 'data_for_damd.json'

        self.dbs = {
            'attraction': self.data_prefix + '/db/attraction_db.json',
            'hospital': self.data_prefix + '/db/hospital_db.json',
            'hotel': self.data_prefix + '/db/hotel_db.json',
            'police': self.data_prefix + '/db/police_db.json',
            'restaurant': self.data_prefix + '/db/restaurant_db.json',
            'taxi': self.data_prefix + '/db/taxi_db.json',
            'train': self.data_prefix + '/db/train_db.json',
        }

        self.dbs_processed = {
            'attraction': self.data_prefix + '/db/attraction_db_processed.json',
            'hospital': self.data_prefix + '/db/hospital_db_processed.json',
            'hotel': self.data_prefix + '/db/hotel_db_processed.json',
            'police': self.data_prefix + '/db/police_db_processed.json',
            'restaurant': self.data_prefix + '/db/restaurant_db_processed.json',
            'taxi': self.data_prefix + '/db/taxi_db_processed.json',
            'train': self.data_prefix + '/db/train_db_processed.json',
        }

        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi

        self.domain_file_path = self.data_prefix + '/multi-woz-processed/domain_files.json'
        self.slot_value_set_path = self.data_prefix + '/db/value_set_processed.json'
        self.vocab_size = 3000