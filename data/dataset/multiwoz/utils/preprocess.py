'''
@Qinyuan Cheng
'''

import os
import re
import json
import copy
import spacy
import argparse
from collections import OrderedDict
from tqdm import tqdm

import utils
import ontology
from config import Config
from db_ops import MultiWozDB
from clean_dataset import clean_slot_values, clean_text

def get_value_set(data_path, all_domains):
    '''
    Get all slots and corresponding values.
    '''
    print('Getting value set...')

    value_set_path = os.path.join(data_path, 'db', 'value_set.json')
    dp_postfix = '_db.json'
    value_set = {}

    for domain in tqdm(all_domains):
        value_set[domain] = {}
        db_file_path = os.path.join(data_path, 'db', domain + dp_postfix)
        with open(db_file_path, 'r') as f:
            db_table = json.loads(f.read())
        
        for record in db_table:
            for slot_name, slot_value in record.items():
                if slot_name in ['id', 'phone', 'location']:
                    continue

                if slot_name.startswith('taxi_'):
                    slot_name = slot_name[5:]
                    value_set[domain][slot_name] = slot_value
                else:
                    if slot_name not in value_set[domain]:
                        value_set[domain][slot_name] = set()
                    if slot_name == 'price' and domain == 'hotel':
                        for key, value in slot_value.items():
                            value_set[domain][slot_name].add(value)
                    else:
                        value_set[domain][slot_name].add(slot_value)

    for domain, domain_information in value_set.items():
        for slot_name in domain_information:
            domain_information[slot_name] = list(domain_information[slot_name])

    with open(value_set_path, 'w') as f:
        json.dump(value_set, f)

    print("Value set is created.")

def get_db_values(data_path, value_set_path, ontology_path, version):
    processed = {}
    bspn_word = set()
    nlp = spacy.load('en_core_web_sm')

    with open(value_set_path, 'r') as f:
        value_set = json.loads(f.read().lower())

    with open(ontology_path, 'r') as f:
        otlg = json.loads(f.read().lower())
    
    for domain, slots in tqdm(value_set.items(), desc='Checking slot values'):
        processed[domain] = {}
        bspn_word.add('[' + domain + ']')
        for slot_name, values in slots.items():
            slot_name = ontology.normlize_slot_names.get(slot_name, slot_name)
            if slot_name in ontology.informable_slots[domain]:
                bspn_word.add(slot_name)
                processed[domain][slot_name] = []
                for value in values:
                    _, value = clean_slot_values(domain, slot_name, value)
                    value = ' '.join([token.text for token in nlp(value)]).strip()
                    processed[domain][slot_name].append(value)
                    for x in value.split():
                        bspn_word.add(x)

    if version == '2.0':
        for domain_slot, values in tqdm(otlg.items(), desc='Checking ontology'): # split domain-slots to domains and slots
            domain, slot = domain_slot.split('-')
            if domain == 'bus':
                domain = 'taxi'
            if slot == 'price range':
                slot = 'pricerange'
            if slot == 'book stay':
                slot = 'stay'
            if slot == 'book day':
                slot = 'day'
            if slot == 'book people':
                slot = 'people'
            if slot == 'book time':
                slot = 'time'
            if slot == 'arrive by':
                slot = 'arrive'
            if slot == 'leave at':
                slot = 'leave'
            if slot == 'leaveat':
                slot = 'leave'
            if slot not in processed[domain]: # add all slots and words of values if not already in processed and bspn_word
                processed[domain][slot] = []
                bspn_word.add(slot)
            for value in values:
                _, value = clean_slot_values(domain, slot, value)
                value = ' '.join([token.text for token in nlp(value)]).strip()
                if value not in processed[domain][slot]:
                    processed[domain][slot].append(value)
                    for x in value.split():
                        bspn_word.add(x)
    elif version == '2.1':
        for domain_slot, values in otlg.items(): # split domain-slots to domains and slots
            tokens = domain_slot.split('-')
            domain = tokens[0]
            slot = tokens[2].lower()
            if slot == 'leaveat':
                slot = 'leave'
            if slot == 'arriveby':
                slot = 'arrive'
            if domain == 'bus':
                domain = 'taxi'
            if slot not in processed[domain]: # add all slots and words of values if not already in processed and bspn_word
                processed[domain][slot] = []
                bspn_word.add(slot)
            for value in values:
                _, value = clean_slot_values(domain, slot, value)
                value = ' '.join([token.text for token in nlp(value)]).strip()
                if value not in processed[domain][slot]:
                    processed[domain][slot].append(value)
                    for x in value.split():
                        bspn_word.add(x)
    elif version == '2.2':
        for domain_slot, values in otlg.items():  # split domain-slots to domains and slots
            tokens = domain_slot.split('-')
            domain = tokens[0]
            slot = tokens[1].lower()
            if domain == 'bus':
                domain = 'taxi'
            if slot not in processed[domain]:  # add all slots and words of values if not already in processed and bspn_word
                processed[domain][slot] = []
                bspn_word.add(slot)
            for value in values:
                _, value = clean_slot_values(domain, slot, value)
                value = ' '.join([token.text for token in nlp(value)]).strip()
                if value not in processed[domain][slot]:
                    processed[domain][slot].append(value)
                    for x in value.split():
                        bspn_word.add(x)


    bspn_word = list(bspn_word)
    with open(value_set_path.replace('.json', '_processed.json'), 'w') as f:
        json.dump(processed, f, indent=2) # save processed.json 
    with open(os.path.join(data_path, 'multi-woz-processed/bspn_word_collection.json'), 'w') as f:
        json.dump(bspn_word, f, indent=2) # save bspn_word

    print('DB value set processed! ')

def preprocess_db(db_paths): # apply clean_slot_values to all dbs
    dbs = {}
    nlp = spacy.load('en_core_web_sm')
    for domain in ontology.all_domains:
        with open(db_paths[domain], 'r') as f: # for every db_domain, read json file 
            dbs[domain] = json.loads(f.read().lower())
            for idx in tqdm(range(len(dbs[domain])), desc='Domain: {:s}'.format(domain)):
                new_entry = copy.deepcopy(dbs[domain][idx])
                for key, value in dbs[domain][idx].items(): # key = slot 
                    if type(value) is not str:
                        continue
                    del new_entry[key]
                    key, value = clean_slot_values(domain, key, value)
                    tokenize_and_back = ' '.join([token.text for token in nlp(value)]).strip()
                    new_entry[key] = tokenize_and_back
                dbs[domain][idx] = new_entry
        with open(db_paths[domain].replace('.json', '_processed.json'), 'w') as f:
            json.dump(dbs[domain], f, indent=2)
        print('[%s] DB processed! '%domain)

class DataPreprocessor(object):
    def __init__(self, cfg):
        self.version = cfg.version
        self.data_prefix = cfg.data_prefix
        self.nlp = spacy.load('en_core_web_sm')
        self.db = MultiWozDB(cfg.dbs) # load all processed dbs
        self.convlab_data = json.loads(open(cfg.data_path).read().lower())
        self.delex_sg_valdict_path = os.path.join(self.data_prefix, 'multi-woz-processed/delex_single_valdict.json')
        self.delex_mt_valdict_path = os.path.join(self.data_prefix, 'multi-woz-processed/delex_multi_valdict.json')
        self.ambiguous_val_path = os.path.join(self.data_prefix, 'multi-woz-processed/ambiguous_values.json')
        self.delex_refs_path = os.path.join(self.data_prefix, 'multi-woz-processed/reference_no.json')
        self.delex_refs = json.loads(open(self.delex_refs_path, 'r').read())
        if not os.path.exists(self.delex_sg_valdict_path):
            self.delex_sg_valdict, self.delex_mt_valdict, self.ambiguous_vals = self.get_delex_valdict()
        else:
            self.delex_sg_valdict = json.loads(open(self.delex_sg_valdict_path, 'r').read())
            self.delex_mt_valdict = json.loads(open(self.delex_mt_valdict_path, 'r').read())
            self.ambiguous_vals = json.loads(open(self.ambiguous_val_path, 'r').read())

        self.vocab = utils.Vocab(cfg.vocab_size)

    def char2word(self, str, begin, end):
        def findend(l, end):
            for i in l:
                if i > end:
                    return i
        def findbegin(l, begin):
            for i in range(len(l)):
                if l[i] > begin:
                    return l[i - 1]
            return l[-1]
        u = str.split(" ")
        begin_list = []
        end_list = []
        for i in range(len(u)):
            if i == 0:
                begin_list.append(0)
                end_list.append(len(u[0]))
            else:
                begin_list.append(1 + end_list[i - 1])
                end_list.append(1 + end_list[i - 1] + len(u[i]))
        # print(begin_list, end_list)
        if begin_list.count(begin) > 0 and end_list.count(end) > 0:
            return begin_list.index(begin), end_list.index(end)
        elif begin_list.count(begin) > 0 and end_list.count(end) == 0:
            return begin_list.index(begin), end_list.index(findend(end_list, end))
        elif begin_list.count(begin) == 0 and end_list.count(end) > 0:
            return begin_list.index(findbegin(begin_list, begin)), end_list.index(end)
        else:
            return begin_list.index(findbegin(begin_list, begin)), end_list.index(findend(end_list, end))

    def delex_by_annotation(self, dial_turn):
        if self.version == '2.1':
            dial_turn['text'] = ' '.join([t.text for t in self.nlp(dial_turn['text'])])
            dial_turn['text'] = ' '.join(dial_turn['text'].replace('.', ' . ').split())

        u = dial_turn['text'].split()
        span = dial_turn['span_info']
        for s in span:
            slot = s[1]
            if slot == 'open':
                continue
            if ontology.da_abbr_to_slot_name.get(slot):
                slot = ontology.da_abbr_to_slot_name[slot]
            for idx in range(s[3], s[4]+1):
                u[idx] = ''
            try:
                u[s[3]] = '[value_'+slot+']'
            except:
                u[5] = '[value_'+slot+']'
        u_delex = ' '.join([t for t in u if t is not ''])
        u_delex = u_delex.replace('[value_address] , [value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_name] [value_name]', '[value_name]')
        u_delex = u_delex.replace('[value_name]([value_phone] )', '[value_name] ( [value_phone] )')
        return u_delex


    def delex_by_valdict(self, text):
        text = clean_text(text)

        text = re.sub(r'\d{5}\s?\d{5,7}', '[value_phone]', text)
        text = re.sub(r'\d[\s-]stars?', '[value_stars]', text)
        text = re.sub(r'\$\d+|\$?\d+.?(\d+)?\s(pounds?|gbps?)', '[value_price]', text)
        text = re.sub(r'tr[\d]{4}', '[value_id]', text)
        text = re.sub(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', '[value_postcode]', text)

        for value, slot in self.delex_mt_valdict.items():
            text = text.replace(value, '[value_%s]'%slot)

        for value, slot in self.delex_sg_valdict.items():
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value:
                    tokens[idx] = '[value_%s]'%slot
            text = ' '.join(tokens)

        for ambg_ent in self.ambiguous_vals:
            start_idx = text.find(' '+ambg_ent)   # ely is a place, but appears in words like moderately
            if start_idx == -1:
                continue
            front_words = text[:start_idx].split()
            ent_type = 'time' if ':' in ambg_ent else 'place'

            for fw in front_words[::-1]:
                if fw in ['arrive', 'arrives', 'arrived', 'arriving', 'arrival', 'destination', 'there', 'reach',  'to', 'by', 'before']:
                    slot = '[value_arrive]' if ent_type=='time' else '[value_destination]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)
                elif fw in ['leave', 'leaves', 'leaving', 'depart', 'departs', 'departing', 'departure',
                                'from', 'after', 'pulls']:
                    slot = '[value_leave]' if ent_type=='time' else '[value_departure]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)

        text = text.replace('[value_car] [value_car]', '[value_car]')
        return text


    def get_delex_valdict(self, ):
        skip_entry_type = {
            'taxi': ['taxi_phone'],
            'police': ['id'],
            'hospital': ['id'],
            'hotel': ['id', 'location', 'internet', 'parking', 'takesbookings', 'stars', 'price', 'n', 'postcode', 'phone'],
            'attraction': ['id', 'location', 'pricerange', 'price', 'openhours', 'postcode', 'phone'],
            'train': ['price', 'id'],
            'restaurant': ['id', 'location', 'introduction', 'signature', 'type', 'postcode', 'phone'],
        }
        entity_value_to_slot= {}
        ambiguous_entities = []
        for domain, db_data in self.db.dbs.items():
            print('Processing entity values in [%s]'%domain)
            if domain != 'taxi':
                for db_entry in db_data:
                    for slot, value in db_entry.items():
                        if slot not in skip_entry_type[domain]:
                            if type(value) is not str:
                                raise TypeError("value '%s' in domain '%s' should be rechecked"%(slot, domain))
                            else:
                                slot, value = clean_slot_values(domain, slot, value)
                                value = ' '.join([token.text for token in self.nlp(value)]).strip()
                                if value in entity_value_to_slot and entity_value_to_slot[value] != slot:
                                    # print(value, ": ",entity_value_to_slot[value], slot)
                                    ambiguous_entities.append(value)
                                entity_value_to_slot[value] = slot
            else:   # taxi db specific
                db_entry = db_data[0]
                for slot, ent_list in db_entry.items():
                    if slot not in skip_entry_type[domain]:
                        for ent in ent_list:
                            entity_value_to_slot[ent] = 'car'
        ambiguous_entities = set(ambiguous_entities)
        ambiguous_entities.remove('cambridge')
        ambiguous_entities = list(ambiguous_entities)
        for amb_ent in ambiguous_entities:   # departure or destination? arrive time or leave time?
            entity_value_to_slot.pop(amb_ent)
        entity_value_to_slot['parkside'] = 'address'
        entity_value_to_slot['parkside, cambridge'] = 'address'
        entity_value_to_slot['cambridge belfry'] = 'name'
        entity_value_to_slot['hills road'] = 'address'
        entity_value_to_slot['hills rd'] = 'address'
        entity_value_to_slot['Parkside Police Station'] = 'name'

        single_token_values = {}
        multi_token_values = {}
        for val, slt in entity_value_to_slot.items():
            if val in ['cambridge']:
                continue
            if len(val.split())>1:
                multi_token_values[val] = slt
            else:
                single_token_values[val] = slt

        with open(self.delex_sg_valdict_path, 'w') as f:
            single_token_values = OrderedDict(sorted(single_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(single_token_values, f, indent=2)
            print('single delex value dict saved!')
        with open(self.delex_mt_valdict_path, 'w') as f:
            multi_token_values = OrderedDict(sorted(multi_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(multi_token_values, f, indent=2)
            print('multi delex value dict saved!')
        with open(self.ambiguous_val_path, 'w') as f:
            json.dump(ambiguous_entities, f, indent=2)
            print('ambiguous value dict saved!')

        return single_token_values, multi_token_values, ambiguous_entities

    def preprocess_main(self, save_path=None, is_test=False):
        """
        """
        data = {}
        self.unique_da = {}
        ordered_sysact_dict = {}
        for fn, raw_dial in tqdm(list(self.convlab_data.items())):

            if self.version == '2.1':
                #these conversations have no dialog_act annotation in MultiWOZ 2.1:
                if fn in ['pmul4707.json', 'pmul2245.json', 'pmul4776.json', 'pmul3872.json', 'pmul4859.json']:
                    continue

            compressed_goal = {} # for every dialog, keep track the goal, domains, requests
            dial_domains, dial_reqs = [], []
            for dom, g in raw_dial['goal'].items():
                if dom != 'topic' and dom != 'message' and g:
                    if g.get('reqt'): # request info. eg. postcode/address/phone
                        for i, req_slot in enumerate(g['reqt']): # normalize request slots
                            if ontology.normlize_slot_names.get(req_slot):
                                g['reqt'][i] = ontology.normlize_slot_names[req_slot]
                                dial_reqs.append(g['reqt'][i])
                    compressed_goal[dom] = g 
                    if dom in ontology.all_domains:
                        dial_domains.append(dom)

            dial_reqs = list(set(dial_reqs))

            dial = {'goal': compressed_goal, 'log': []}
            single_turn = {}
            constraint_dict = OrderedDict()
            prev_constraint_dict = {}
            prev_turn_domain = ['general']
            ordered_sysact_dict[fn] = {}

            for turn_num, dial_turn in enumerate(raw_dial['log']):
                # for user turn, have text
                # sys turn: text, belief states(metadata), dialog_act, span_info
                dial_state = dial_turn['metadata']
                if self.version=="2.2":
                    if dial_turn['span_info'] != []:
                        for i in range(len(dial_turn['span_info'])):
                            dial_turn['span_info'][i][3], dial_turn['span_info'][i][4] = self.char2word(dial_turn['text'], dial_turn['span_info'][i][3], dial_turn['span_info'][i][4])
                    dial_turn['text'] = ' '.join([t.text for t in self.nlp(dial_turn['text'])])
                    dial_turn["text"] = ' '.join(dial_turn["text"].replace(".", " . ").split())
                if not dial_state:   # user
                    # delexicalize user utterance, either by annotation or by val_dict
                    u = ' '.join(clean_text(dial_turn['text']).split())
                    if dial_turn['span_info']:
                        u_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        u_delex = self.delex_by_valdict(dial_turn['text'])

                    single_turn['user'] = u
                    single_turn['user_delex'] = u_delex

                else:   # system
                    # delexicalize system response, either by annotation or by val_dict
                    if dial_turn['span_info']:
                        s_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        if not dial_turn['text']:
                            print(fn)
                        s_delex = self.delex_by_valdict(dial_turn['text'])
                    single_turn['resp'] = s_delex
                    if self.version=="2.2":
                        single_turn['nodelx_resp'] = ' '.join(clean_text(dial_turn['text']).split())
                    # get belief state, semi=informable/book=requestable, put into constraint_dict
                    for domain in dial_domains:
                        if not constraint_dict.get(domain):
                            constraint_dict[domain] = OrderedDict()
                        info_sv = dial_state[domain]['semi']
                        for s,v in info_sv.items():
                            if self.version=="2.2":
                                if v == []:
                                    v = ""
                                elif type(v) == list:
                                    v = v[0]
                            s,v = clean_slot_values(domain, s,v)
                            if len(v.split())>1:
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v
                        book_sv = dial_state[domain]['book']
                        for s,v in book_sv.items():
                            if s == 'booked':
                                continue
                            if self.version=="2.2":
                                if v == []:
                                    v = ""
                                elif type(v) == list:
                                    v = v[0]
                            s,v = clean_slot_values(domain, s,v)
                            if len(v.split())>1:
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v

                    constraints = [] # list in format of [domain] slot value
                    cons_delex = []
                    turn_dom_bs = []
                    for domain, info_slots in constraint_dict.items():
                        if info_slots:
                            constraints.append('['+domain+']')
                            cons_delex.append('['+domain+']')
                            for slot, value in info_slots.items():
                                constraints.append(slot)
                                constraints.extend(value.split())
                                cons_delex.append(slot)
                            if domain not in prev_constraint_dict:
                                turn_dom_bs.append(domain)
                            elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                turn_dom_bs.append(domain)


                    sys_act_dict = {}
                    turn_dom_da = set()
                    for act in dial_turn['dialog_act']:
                        d, a = act.split('-') # split domain-act
                        turn_dom_da.add(d)
                    turn_dom_da = list(turn_dom_da)
                    if len(turn_dom_da) != 1 and 'general' in turn_dom_da:
                        turn_dom_da.remove('general')
                    if len(turn_dom_da) != 1 and 'booking' in turn_dom_da:
                        turn_dom_da.remove('booking')

                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]

                    # get system action
                    for dom in turn_domain:
                        sys_act_dict[dom] = {}
                    add_to_last_collect = []
                    booking_act_map = {'inform': 'offerbook', 'book': 'offerbooked'}
                    for act, params in dial_turn['dialog_act'].items():
                        if act == 'general-greet':
                            continue
                        d, a = act.split('-')
                        if d == 'general' and d not in sys_act_dict:
                            sys_act_dict[d] = {}
                        if d == 'booking':
                            d = turn_domain[0]
                            a = booking_act_map.get(a, a)
                        add_p = []
                        for param in params:
                            p = param[0]
                            if p == 'none':
                                continue
                            elif ontology.da_abbr_to_slot_name.get(p):
                                p = ontology.da_abbr_to_slot_name[p]
                            if p not in add_p:
                                add_p.append(p)
                        add_to_last = True if a in ['request', 'reqmore', 'bye', 'offerbook'] else False
                        if add_to_last:
                            add_to_last_collect.append((d,a,add_p))
                        else:
                            sys_act_dict[d][a] = add_p
                    for d, a, add_p in add_to_last_collect:
                        sys_act_dict[d][a] = add_p

                    for d in copy.copy(sys_act_dict):
                        acts = sys_act_dict[d]
                        if not acts:
                            del sys_act_dict[d]
                        if 'inform' in acts and 'offerbooked' in acts:
                            for s in sys_act_dict[d]['inform']:
                                sys_act_dict[d]['offerbooked'].append(s)
                            del sys_act_dict[d]['inform']


                    ordered_sysact_dict[fn][len(dial['log'])] = sys_act_dict

                    sys_act = []
                    if 'general-greet' in dial_turn['dialog_act']:
                        sys_act.extend(['[general]', '[greet]'])
                    for d, acts in sys_act_dict.items():
                        sys_act += ['[' + d + ']']
                        for a, slots in acts.items():
                            self.unique_da[d+'-'+a] = 1
                            sys_act += ['[' + a + ']']
                            sys_act += slots


                    # get db pointers
                    matnums = self.db.get_match_num(constraint_dict)
                    match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
                    match = matnums[match_dom]
                    dbvec = self.db.addDBPointer(match_dom, match)
                    bkvec = self.db.addBookingPointer(dial_turn['dialog_act'])

                    single_turn['pointer'] = ','.join([str(d) for d in dbvec + bkvec]) # 4 database pointer for domains, 2 for booking
                    single_turn['match'] = str(match)
                    single_turn['constraint'] = ' '.join(constraints)
                    single_turn['cons_delex'] = ' '.join(cons_delex)
                    single_turn['sys_act'] = ' '.join(sys_act)
                    single_turn['turn_num'] = len(dial['log'])
                    single_turn['turn_domain'] = ' '.join(['['+d+']' for d in turn_domain])

                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)

                    if 'user' in single_turn:
                        dial['log'].append(single_turn)
                        for t in single_turn['user'].split() + single_turn['resp'].split() + constraints + sys_act:
                            self.vocab.add_word(t)
                        for t in single_turn['user_delex'].split():
                            if '[' in t and ']' in t and not t.startswith('[') and not t.endswith(']'):
                                single_turn['user_delex'].replace(t, t[t.index('['): t.index(']')+1])
                            elif not self.vocab.has_word(t):
                                self.vocab.add_word(t)

                    single_turn = {}
            data[fn] = dial

        self.vocab.construct()
        self.vocab.save_vocab(os.path.join(self.data_prefix, 'multi-woz-processed/vocab'))
        with open(os.path.join(self.data_prefix, 'multi-woz-analysis/dialog_acts.json'), 'w') as f:
            json.dump(ordered_sysact_dict, f, indent=2)
        with open(os.path.join(self.data_prefix, 'multi-woz-analysis/dialog_act_type.json'), 'w') as f:
            json.dump(self.unique_da, f, indent=2)
        return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument for preprocessing")
    parser.add_argument("--version", type=str, required=True, choices=["2.0", "2.1", "2.2"])
    args = parser.parse_args()

    cfg = Config(args.version)

    get_value_set(cfg.data_prefix, ontology.all_domains)
    get_db_values(cfg.data_prefix, cfg.value_set_path, cfg.ontology_path, args.version)
    preprocess_db(cfg.dbs)
    data_preprocessor = DataPreprocessor(cfg)
    data = data_preprocessor.preprocess_main()
    if not os.path.exists(cfg.processed_data_path):
        os.mkdir(cfg.processed_data_path)

    with open(os.path.join(cfg.processed_data_path, cfg.data_for_damd), 'w') as f:
        json.dump(data, f, indent=2)