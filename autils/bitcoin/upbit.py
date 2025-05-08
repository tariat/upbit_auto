UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_ACCESS_KEY = os.getenv("UPBIT_SECRET_KEY")

from __upbit__ import UpBit
api = UpBit(access_key=UPBIT_ACCESS_KEY, secret_key=UPBIT_ACCESS_KEY)
