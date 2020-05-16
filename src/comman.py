import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Constant:
    LOG_PATH = os.path.join(BASE_PATH, 'log')
    DATA_PATH = os.path.join(BASE_PATH, 'data/input')
    OUTPUT_PATH = os.path.join(BASE_PATH, 'data/output')
    DATA_FILE_NAME='data_9000.xlsx'
    DATA_OUTPUT_FILE_NAME='processed_data.xlsx'
    STOP_WORDS_PATH = '../venv/lib/python3.6/site-packages/hazm/data/stopwords.dat'
    STOP_WORDS = ['دیگران', 'همچنان', 'مدت', 'چیز', 'سایر', 'جا', 'طی', 'کل', 'کنونی', 'بیرون', 'مثلا', 'کامل', 'کاملا', 'آنکه', 'موارد', 'واقعی',
              'امور', 'اکنون', 'بطور', 'بخشی', 'تحت', 'چگونه', 'عدم', 'نوعی', 'حاضر', 'وضع', 'مقابل', 'کنار', 'خویش', 'نگاه', 'درون', 'زمانی',
              'بنابراین', 'تو', 'خیلی', 'بزرگ', 'خودش', 'جز', 'اینجا', 'مختلف', 'توسط', 'نوع', 'همچنین', 'آنجا', 'قبل', 'جناح', 'اینها', 'طور',
              'شاید', 'ایشان', 'جهت', 'طریق', 'مانند', 'پیدا', 'ممکن', 'کسانی', 'جای', 'کسی', 'غیر', 'بی', 'قابل', 'درباره', 'جدید', 'وقتی', 'اخیر',
              'چرا', 'بیش', 'روی', 'طرف', 'جریان', 'زیر', 'آنچه', 'البته', 'فقط', 'چیزی', 'چون', 'برابر', 'هنوز', 'بخش', 'زمینه', 'بین', 'بدون',
              'استفاده', 'همان', 'نشان', 'بسیاری', 'بعد', 'عمل', 'روز', 'اعلام', 'چند', 'آنان', 'بلکه', 'امروز', 'تمام', 'بیشتر', 'آیا', 'برخی', 'علیه',
              'دیگری', 'ویژه', 'گذشته', 'انجام', 'حتی', 'داده', 'راه', 'سوی', 'ولی', 'زمان', 'حال', 'تنها', 'بسیار', 'یعنی', 'عنوان', 'همین', 'هبچ',
              'پیش', 'وی', 'یکی', 'اینکه', 'وجود', 'شما', 'پس', 'چنین', 'میان', 'مورد', 'چه', 'اگر', 'همه', 'نه', 'دیگر', 'آنها', 'باید', 'هر', 'او',
              'ما', 'من', 'تا', 'نیز', 'اما', 'یک', 'خود', 'بر', 'یا', 'هم', 'را', 'این', 'با', 'آن', 'برای', 'و', 'در', 'به', 'که', 'از'
              'کن', 'کرد', 'کردن', 'باش', 'بود', 'بودن', 'شو', 'شد', 'شدن', 'ددار', 'داشت', 'داشتن', 'خواه', 'خواست', 'خواستن', 'گوی', 'گفت',
              'گفتن', 'گرفت', 'گرفتن', 'آمد', 'آمدن', 'توانست', 'توانستن', 'یافت', 'یافتن', 'آورد', 'آوردن','هرگز','نمي کند', 'است','هستند','با','از','چه','باشد',
              'مي کنند']
    TEST_SIZE = 0.30
    SPLIT_DATA = True
    RANDOM_STATE = 0
