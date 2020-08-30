import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile
from pandas import DataFrame
from utils.gmail.transmitter import GMailTransmitter

SERVICE_ACCOUNT = 'yuu.meiji.research@gmail.com'
PASSWORD = 'risuka07S;'


class TestGMail(TestCase):
    transmitter = GMailTransmitter(token_path='token.pickle', secret_path='client_secret.json',
                                   account_json='account.json')

    def test_basic(self) -> None:
        self.transmitter.send(subject='[Test] Basic',
                              body='This is the test mail for check only a subject and a body.',
                              to='yuu.meiji.gkb@gmail.com')

    def test_with_single_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_file = os.path.join(tmp_dir, f'sample.csv')
            DataFrame([['a', 'b'], [1., 4.]], columns=['str', 'float']).to_csv(csv_file, index=False)

            self.transmitter.send(subject='[Test] with file',
                                  body='This is the test mail for sending the mail with some files.',
                                  to='yuu.meiji.gkb@gmail.com',
                                  file_path=csv_file,
                                  extension='csv')

    def test_with_several_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_file = os.path.join(tmp_dir, f'sample.csv')
            DataFrame([['a', 'b'], [1., 4.]], columns=['str', 'float']).to_csv(csv_file, index=False)

            zip_file = os.path.join(tmp_dir, 'sample.zip')
            with ZipFile(zip_file, 'w') as zf:
                zf.write(csv_file, arcname=csv_file)

            self.transmitter.send(subject='[Test] with file',
                                  body='This is the test mail for sending the mail with some files.',
                                  to='yuu.meiji.gkb@gmail.com',
                                  file_path=(csv_file, zip_file),
                                  extension=('csv', 'zip'))
