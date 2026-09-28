import hashlib, tempfile, unittest
from pathlib import Path
from verify_dependencies import verify

class VerifyTest(unittest.TestCase):
    def test_original_same_size_mutation_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'dependency';p.write_bytes(b'original')
            m={'files':[{'path':str(p),'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}]}
            self.assertTrue(verify(m)['success'])
            p.write_bytes(b'changed!')
            self.assertEqual(verify(m)['failures'][0]['reason'],'sha256')
            p.unlink()
            self.assertEqual(verify(m)['failures'][0]['reason'],'missing')

if __name__=='__main__':unittest.main()
