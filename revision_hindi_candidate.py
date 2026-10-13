"""Post-diagnostic candidate; isolated from both frozen V8 stages."""
import revision_v7_prompts as frozen

VERSION = 'hindi-alternative-2-single-line-candidate-1'
FORMAT_CLARIFICATION = (
    'उत्तर में चुने गए व्यक्तियों की ID दें, मित्रता की जोड़ियाँ नहीं। '
    'पूरे उत्तर में केवल एक पंक्ति हो; नई पंक्ति न जोड़ें। '
    'एक व्यक्ति चुनने पर केवल उसकी ID दें; एक से अधिक व्यक्ति चुनने पर '
    'उनकी अलग-अलग ID एक ही पंक्ति में अल्पविराम से अलग करके दें।'
)


def hindi_alternative_2(country, actor, count, *, retry=False):
    """Keep the original treatment; clarify serialization, not friend selection."""
    if type(count) is not int or not 1 <= count <= 49:
        raise ValueError('Expected 1 to 49 nominations for the 50-person roster.')
    render = frozen.treatment_retry if retry else frozen.treatment_prompt
    return (render('local', 'hindi', country, actor, count, wording='alternative_2')
            + ' ' + FORMAT_CLARIFICATION)
