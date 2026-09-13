<?php
gc_disable();

require 'vendor/autoload.php';

use Gregwar\Captcha\PhraseBuilder;
use Gregwar\Captcha\CaptchaBuilder;

if ($argc < 2) {
    exit(1);
}

$tot = intval($argv[1]);
$seed_offset = isset($argv[2]) ? intval($argv[2]) : 0;
mt_srand((int)(microtime(true) * 1000) + $seed_offset + getmypid());

$ostream = fopen("php://stdout", "wb");

$phraseBuilder = new PhraseBuilder(4);
$builder = new CaptchaBuilder(null, $phraseBuilder);

for ($i = 0; $i < $tot; ++$i) {
    $builder->setPhrase($phraseBuilder->build(4));
    $builder->build(90, 35);
    $phrase = $builder->getPhrase();
    
    $jpeg_data = $builder->get(75);
    $len = strlen($jpeg_data);

    $header = pack("n", $len) . $phrase;
    
    fwrite($ostream, $header, 6);
    fwrite($ostream, $jpeg_data, $len);
    if (PHP_VERSION_ID < 80000) {
        imagedestroy($builder->getContents());
    }

    if ($i % 1000 === 0) {
        fflush($ostream);
    }
}

fflush($ostream);
fclose($ostream);
