from encrypt import decrypt
import itertools
from multiprocessing import Pool
import time 
from datetime import timedelta
import smtplib
from email.message import EmailMessage
import string

with open("server/templates/secret.html.crypt", "rb") as fp:
    ciphertext = fp.read()



def decrypt_target(password):
    try:
        plaintext = decrypt(password, ciphertext)

    except Exception as e:
        return None

    print(password)
    return plaintext

def testing():

    print("Testing password: password")
    result = decrypt_target("password1234")

    if result is not None:
        print(result)


def main():

    #oldpassword = "aw2246french44pledgeor"
    #ciphertextloc = "server/templates/secret.html.crypt"

    oldpassword = "aw2246french44pledgeor"
    ciphertextloc = "server/templates/secret.html.crypt"

    puncts = string.punctuation
    digits = string.digits

    alphabet = puncts + digits + string.ascii_letters



    # oldpassword.join(t) was wrong: str.join() inserts its string BETWEEN
    # the elements of the iterable, so e.g. "password".join(('1','2','3','4'))
    # gives '1password2password3password4' -- three copies of oldpassword
    # woven between the digits -- and for repeat=1 there's nothing to insert
    # a separator between at all, so oldpassword vanishes and you just get
    # the bare character back. "password1234" was never reachable from any
    # of these, for any tuple. What we actually want is oldpassword as a
    # fixed prefix, followed by the suffix characters concatenated plainly:
    # ''.join(t) turns the tuple into a string, then + appends it after
    # oldpassword exactly once.
    suffix1 = (oldpassword + ''.join(t) for t in itertools.product(alphabet, repeat=5))
    suffix2 = (oldpassword + ''.join(t) for t in itertools.product(alphabet, repeat=2))
    suffix3 = (oldpassword + ''.join(t) for t in itertools.product(alphabet, repeat=3))
    suffix4 = (oldpassword + ''.join(t) for t in itertools.product(alphabet, repeat=4))

    total = sum(len(alphabet) ** k for k in range(1,5))
    all_candidates = itertools.chain(suffix1, suffix2, suffix3, suffix4)
    start = time.perf_counter()
    last_print = start
    print_interval = 5  # seconds between progress prints
    count = 0
    with Pool(processes=12) as pool:
            for result in pool.imap_unordered(decrypt_target, all_candidates, chunksize=128):
                #print(result)
                count += 1
                now = time.perf_counter()
                if now - last_print >= print_interval:
                    elapsed = now - start
                    rate = count / elapsed                      # candidates per second
                    remaining = total - count
                    eta_seconds = remaining / rate
                    print(f"{count}/{total} ({100*count/total:.2f}%) "f"- {rate:.1f}/s - ETA {timedelta(seconds=int(eta_seconds))}")
                    last_print = now
    
                if result is not None:
                    print(f"SUCCESS!  Secret is: {result}")
                    #fp.write(f"SUCCESS!  Password is: {result}")
                    #fp.close()
                    pool.terminate()
                    
                    #log_job(record)
                    exit(0)
    
    print('FAILURE!')
       #record = {"script": scriptname, "description": jobdescription, "count": total, "result": f'Failure'}
        #log_job(record)
        #body = json.dumps(record, indent=2)
        #send_notification('Hashing Failure!', body)

    


    







if __name__ == '__main__':
    main()
