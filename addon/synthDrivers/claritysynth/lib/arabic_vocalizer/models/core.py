from pathlib import Path
from typing import Literal, List, Union

from ..models import Shakkala, Shakkelha

_MODEL_TYPE = Literal['shakkala', 'shakkelha']


def _find_model_file(fname):
    """Locate a diacritizer model file, checking the persistent ClaritySynth
    download dir first (models are no longer bundled but work if downloaded),
    then this package's own data/ folder."""
    import os
    # persistent user data dir, via the driver's data_paths if reachable
    try:
        import globalVars
        base = os.path.join(globalVars.appArgs.configPath, "claritysynth")
        for sub in (("tts_arabic", "data"), ("vowelizers",), ()):
            cand = os.path.join(base, *sub, fname)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    pkg = Path(__file__).parent.parent.joinpath('data').joinpath(fname)
    return pkg.as_posix()


def get_model(model: _MODEL_TYPE = 'shakkelha'):
    assert model in ('shakkala', 'shakkelha')
    if model == 'shakkala':
        return Shakkala(sd_path=_find_model_file('shakkala.onnx'))
    elif model == 'shakkelha':
        return Shakkelha(sd_path=_find_model_file('shakkelha.onnx'))


def vocalize(input_text: Union[str, List[str]], 
             model: _MODEL_TYPE = 'shakkelha',
             return_probs: bool = False
             ) -> Union[str, List[str]]:
    """
    Parameters:
        input_text (str|list[str]): Unvocalized text
        model: Vocalization model [shakkala|shakkelha]
        return_probs: Return probabilities?
        
    Returns:
        (str|list[str]): Predicted vocalized text
    
    Examples:
        >>> from arabic_vocalizer import vocalize
        >>> input_text = "اللغة العربية هي أكثر اللغات السامية تحدثا، وإحدى أكثر اللغات انتشارا في العالم، يتحدثها أكثر من 467 مليون نسمة"
        # shakkala output
        >>> print(vocalize(input_text, model='shakkala'))
        >>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مَنْ 467 مُلْيُونُ نُسْمَةَ
        # shakkelha output
        >>> print(vocalize(input_text, model='shakkelha'))
        >>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مِنْ 467 مَلْيُونٍ نَسَمَةً

    """
    assert model in ('shakkala', 'shakkelha')
    if not hasattr(vocalize, model):
        setattr(vocalize, model, get_model(model=model))

    if model == 'shakkala':
        return vocalize.shakkala.predict(input_text, return_probs=return_probs)
    elif model == 'shakkelha':
        return vocalize.shakkelha.predict(input_text, return_probs=return_probs)
    else:
        return  