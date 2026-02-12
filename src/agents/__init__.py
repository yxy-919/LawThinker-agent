from .base_agent import Agent


from .trainee_KQ import (
    Trainee_consult,
    GPTTrainee_consult,
    Qwen3_14BTrainee_consult,
    Qwen3_32BTrainee_consult,
    Gemma12BTrainee_consult,
    GLM9BTrainee_consult,
    LawLLMTrainee_consult,
    Deepseekv3Trainee_consult,
    LLaMa3_3Trainee_consult,
    InternLM3Trainee_consult,
    Ministral8BTrainee_consult
)

from .lawyer_CD import (
    Lawyer_generation,
    GPTLawyer_generation,
    Qwen3_14BLawyer_generation,
    Qwen3_32BLawyer_generation,
    Gemma12BLawyer_generation,
    GLM9BLawyer_generation,
    LawLLMLawyer_generation,
    Deepseekv3Lawyer_generation,
    LLaMa3_3Lawyer_generation,
    InternLM3Lawyer_generation,
    Ministral8BLawyer_generation
)

from .lawyer_DD import (
    Lawyer_generation,
    GPTLawyer_generation,
    Qwen3_14BLawyer_generation,
    Qwen3_32BLawyer_generation,
    Gemma12BLawyer_generation,
    GLM9BLawyer_generation,
    Chatlaw2Lawyer_generation,
    LawLLMLawyer_generation,
    Deepseekv3Lawyer_generation,
    LLaMa3_3Lawyer_generation,
    InternLM3Lawyer_generation,
    Ministral8BLawyer_generation
)

from .specific_character_CD import (
    Specific_character_generation,
    GPTSpecific_character_generation,
    Qwen332BSpecific_character_generation
)

from .specific_character_DD import (
    Specific_character_generation,
    GPTSpecific_character_generation,
    Qwen332BSpecific_character_generation
)

from .general_public_KQ import (
    Qwen3_32BGeneral_public_consult,
    General_public_consult,
    GPTGeneral_public_consult
)

from .general_public_LC import (
    GPTGeneral_public_consult,
    Qwen3_32BGeneral_public_consult,
    General_public_consult
)

from .trainee_LC import (
    Trainee_consult,
    GPTTrainee_consult,
    Qwen3_14BTrainee_consult,
    Qwen3_32BTrainee_consult,
    Gemma12BTrainee_consult,
    GLM9BTrainee_consult,
    LawLLMTrainee_consult,
    Deepseekv3Trainee_consult,
    LLaMa3_3Trainee_consult,
    InternLM3Trainee_consult,
    Ministral8BTrainee_consult
)

from .judge_CI import (
    Judge_civilPrediction,
    GPTJudge_civilPrediction,
    Qwen3_14BJudge_civilPrediction,
    Qwen3_32BJudge_civilPrediction,
    Gemma12BJudge_civilPrediction,
    GLM9BJudge_civilPrediction,
    LawLLMJudge_civilPrediction,
    Deepseekv3Judge_civilPrediction,
    LLaMa3_3Judge_civilPrediction,
    InternLM3Judge_civilPrediction,
    Ministral8BJudge_civilPrediction
)

from .plaintiff_CI import (
    Plaintiff_civilPrediction,
    GPTPlaintiff_civilPrediction,
    Qwen3_32BPlaintiff_civilPrediction
)

from .defendant_CI import (
    Defendant_civilPrediction,
    GPTDefendant_civilPrediction,
    Qwen3_32BDefendant_civilPrediction
)

from .defendant_CR import (
    Defendant_criminalPrediction,
    GPTDefendant_criminalPrediction,
    Qwen3_32BDefendant_criminalPrediction
)

from .lawyer_CR import (
    Lawyer_criminalPrediction,
    GPTLawyer_criminalPrediction,
    Qwen3_32BLawyer_criminalPrediction
)

from .procurator_CR import (
    Procurator_criminalPrediction,
    GPTProcurator_criminalPrediction,
    Qwen3_32BProcurator_criminalPrediction
)

from .judge_CR import (
    Judge_criminalPrediction,
    GPTJudge_criminalPrediction,
    Qwen3_14BJudge_criminalPrediction,
    Qwen3_32BJudge_criminalPrediction,
    Gemma12BJudge_criminalPrediction,
    GLM9BJudge_criminalPrediction,
    Chatlaw2Judge_criminalPrediction,
    Deepseekv3Judge_criminalPrediction,
    LawLLMJudge_criminalPrediction,
    InternLM3Judge_criminalPrediction,
    LLaMa3_3Judge_criminalPrediction,
    Ministral8BJudge_criminalPrediction
)