import json
import os

def cleanCodeforces(in_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    d = None
    with open(in_path, 'r') as f:
        d = json.load(f)
    assert d is not None 

    users = d['result']

    out_users = {'rating':[], 'maxRating':[]}
    for user in users:
        assert isinstance(user['rating'], int) and isinstance(user['maxRating'], int) 
        out_users['rating'].append(user['rating'])
        out_users['maxRating'].append(user['maxRating'])

    out_users['rating'] = sorted(out_users['rating'], reverse=True)
    out_users['maxRating'] = sorted(out_users['maxRating'], reverse=True)
    with open(out_path, 'w') as f:
        json.dump(out_users, f)
    print('#users: ', len(out_users['rating']))

if __name__ == '__main__':
    current_file_dir = os.path.dirname(__file__)  # folder where this script is located

    competingUsersCF = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'raw', 'codeforces', 'competingUsers.json'))
    cleanCompetingUsersCF = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'clean', 'codeforces', 'competingUsers.json'))

    activeUsersCF = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'raw', 'codeforces', 'activeUsers.json'))
    cleanActiveUsersCF = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'clean', 'codeforces', 'activeUsers.json'))

    allUsersCF = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'raw', 'codeforces', 'fullUsers.json'))
    cleanAllUsersCF = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'dataset', 'clean', 'codeforces', 'fullUsers.json'))

    cleanCodeforces(competingUsersCF, cleanCompetingUsersCF)
    cleanCodeforces(activeUsersCF, cleanActiveUsersCF)
    cleanCodeforces(allUsersCF, cleanAllUsersCF)