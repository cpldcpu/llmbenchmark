# LLM Python Raytracer - Benchmark Experiment

This repository contains various Python raytracer implementations generated by different Large Language Models (LLMs) using a single prompt and one attempt.

## Prompt Used

The following prompt was used for all LLMs:
> Write a raytracer that renders an interesting scene with many colourful lightsources in python. Output a 800x600 image as a png

## Implementations

* [Claude Sonnet 3.5 (new)](raytracer_sonnet.py)
* [Claude Sonnet 3.7](raytracer_sonnet37.py)
* [Claude Sonnet 3.7 Thinking](raytracer_sonnet37_thinking.py)
* [DeepSeek v3](raytracer_DeepSeek_v3.py)
* [DeepSeek v3_0324](raytracer_DeepSeek_v3_0324.py)
* [DeepSeek R1](raytracer_DeepSeek_r1.py)
* [Gemini Flash Thinking](raytracer_gemini_flash_thinking.py)
* [Gemini 2 Flash](raytracer_gemini_2_flash.py)
* [Gemini 2.5 Pro Exp](raytracer_gemini_2_pro_exp.py)
* [Gemini 2.5 Pro 0506](/variance_gemini25pro0506/iteration4.py)
* [Grok 2](raytracer_grok2.py)
* [Grok 3](raytracer_grok3.py)
* [Llama 3.3 70b](raytracer_llama3_3_70b.py)
* [Llama4 Scout](/variance_llama4_scout/iteration4.py)
* [Llama4 Maverick](/variance_llama4_maverick/iteration4.py)
* [O1](raytracer_o1.py)
* [O3-Mini](raytracer_o3-mini.py)
* [O4-Mini High](/variance_o4_mini_high/iteration1.py)
* [GPT-4o](raytracer_4o.py)
* [GPT-4.5](raytracer_gpt4_5.py)
* [Quasar Alpha](/variance_quasar_alpha/iteration1.py)
* [Optimus Alpha](/variance_optimus_alpha/iteration1.py)
* [Qwen Max](raytracer_qwen_max.py)
* [QwQ32B](raytracer_qwq32b.py)

The following LLMs did not return properly functioning code: Grok3-thinking, DeepSeek-R1, Le Chat (Mistral Large 2), 8b models, llama3-405b/70b, llama3-hermes3-405b/70b, Hunyuan-T1,Qwen3-235. Generally, the R1-adjacent reasoning models tend to overthink code fragments and then come up with dysfunctional code. 

## Results

### Performance Notes

* Most implementations take several minutes to render, which is not unexpected for a Python raytracer. (Try C next? :D)
* Performance varies between implementations, with some taking significantly longer to complete
* DeepSeek R1 spent 566 thinking and then outputted a broken file. I was not able to restart due to a busy server.
* 4o's first attempt required corrections, but the second attempt was successful

### Output Images for different LLMs

<table align="center">
  <tr>
    <td align="center">
      <img src="images/sonnet.png" alt="Claude Sonnet 3.5 (new)" width="400" /><br/>
      Claude Sonnet 3.5 (new)
    </td>
    <td align="center">
      <img src="images/sonnet37.png" alt="Claude Sonnet 3.7" width="400" /><br/>
      Claude Sonnet 3.7
    </td>
    <td align="center">
      <img src="images/sonnet37_thinking.png" alt="Claude Sonnet 3.7 Thinking" width="400" /><br/>
      Claude Sonnet 3.7 Thinking
    </td>
    <td align="center">
      <img src="images/gpt-4o.png" alt="GPT-4o" width="400" /><br/>
      GPT-4o
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/deepseek_v3.png" alt="DeepSeek v3" width="400" /><br/>
      DeepSeek v3
    </td>
    <td align="center">
      <img src="images/gemini_flash_thinking.png" alt="Gemini Flash Thinking" width="400" /><br/>
      Gemini Flash Thinking
    </td>
    <td align="center">
      <img src="images/gemini_2_flash.png" alt="Gemini 2 Flash" width="400" /><br/>
      Gemini 2 Flash
    </td>
    <td align="center">
      <img src="images/grok2_raytracer.png" alt="Grok 2" width="400" /><br/>
      Grok 2
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/grok3_raytracer.png" alt="Grok 3" width="400" /><br/>
      Grok 3
    </td>
    <td align="center">
      <img src="images/o1.png" alt="O1" width="400" /><br/>
      O1
    </td>
    <td align="center">
      <img src="images/o3-mini.png" alt="O3-Mini" width="400" /><br/>
      O3-Mini
    </td>
    <td align="center">
      <img src="images/QwenMax.png" alt="Qwen Max" width="400" /><br/>
      Qwen Max
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/gpt-4_5.png" alt="GPT-4.5" width="400" /><br/>
      GPT-4.5
    </td>
    <td align="center">
      <img src="images/QwQ32B.png" alt="QwQ32B" width="400" /><br/>
      QwQ32B
    </td>
    <td align="center">
      <img src="images/codestral.png" alt="Codestral" width="400" /><br/>
      Codestral
    </td>
    <td align="center">
      <img src="images/deepseek_v3_0324.png" alt="DeepSeek_v3_0324" width="400" /><br/>
      Deepseek V3 0324
    </td>
  </tr>
    <tr>
    <td align="center">
      <img src="images/gemini25proexp.png" alt="Gemini 2.5 pro exp" width="400" /><br/>
      Gemini 2.5 pro exp
    </td>
    <td align="center">
      <img src="images/llama3_3_70b.png" alt="llama3_3_70b" width="400" /><br/>
      LLama 3.3 70b
    </td>
    <td align="center">
      <img src="variance_quasar_alpha/raytraced_scene2.png" alt="Quasar Alpha" width="400" /><br/>
      Quasar Alpha
    </td>
    <td align="center">
      <img src="variance_optimus_alpha/raytraced_scene2.png" alt="Optimus Alpha" width="400" /><br/>
      Optimus Alpha
     </td>
  </tr>
</table>

### Variability test

Since the tests above are based on one-shot prompts, they are not necessarily representative of the LLM's average capabilities. I ran the experiment 4 times with [Sonnet-3.5](variance_sonnet35/), [Sonnet-3.7](variance_sonnet37/), [GPT4.5](variance_gpt45/) and [grok3](variance_grok3/) to test for consistency.

We can clearly see a significant change in behavior between the two models. Sonnet-3.5 produces slight variations of a basic scene with red, green and blue spheres, while Sonnet-3.7 uses more objects, colors and more variations in general. GPT 4.5 and Grok3 are also more of a three-sphere llm.

#### Claude Sonnet 3.5 Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_sonnet35/raytraced_scene1.png" alt="Sonnet 3.5 - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_sonnet35/raytraced_scene2.png" alt="Sonnet 3.5 - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_sonnet35/raytraced_scene3.png" alt="Sonnet 3.5 - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_sonnet35/raytraced_scene4.png" alt="Sonnet 3.5 - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Claude Sonnet 3.7 Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_sonnet37/raytraced_scene1.png" alt="Sonnet 3.7 - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_sonnet37/raytraced_scene2.png" alt="Sonnet 3.7 - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_sonnet37/raytraced_scene3.png" alt="Sonnet 3.7 - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_sonnet37/raytraced_scene4.png" alt="Sonnet 3.7 - Test 4" width="400" />
    </td>
  </tr>
</table>

#### DeepSeek V3 (original) Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_deepseek_v3/raytraced_scene1.png" alt="DeepSeek V3 - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_deepseek_v3/raytraced_scene2.png" alt="DeepSeek V3 - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_deepseek_v3/raytraced_scene3.png" alt="DeepSeek V3 - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_deepseek_v3/raytraced_scene4.png" alt="DeepSeek V3 - Test 4" width="400" />
    </td>
  </tr>
</table>

#### DeepSeek V3 (0324) Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_deepseek_v3_0324/raytraced_scene1.png" alt="DeepSeek V3 (0324) - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_deepseek_v3_0324/raytraced_scene2.png" alt="DeepSeek V3 (0324) - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_deepseek_v3_0324/raytraced_scene3.png" alt="DeepSeek V3 (0324) - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_deepseek_v3_0324/raytraced_scene4.png" alt="DeepSeek V3 (0324) - Test 4" width="400" />
    </td>
  </tr>
</table>

#### GPT 4.5 Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_gpt45/raytraced_scene1.png" alt="GPT 4.5 - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_gpt45/raytraced_scene2.png" alt="GPT 4.5 - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_gpt45/raytraced_scene3.png" alt="GPT 4.5 - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_gpt45/raytraced_scene4.png" alt="GPT 4.5 - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Grok3 Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_grok3/raytraced_scene1.png" alt="Grok 3 - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_grok3/raytraced_scene2.png" alt="Grok 3 - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_grok3/raytraced_scene3.png" alt="Grok 3 - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_grok3/raytraced_scene4.png" alt="Grok 3 - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Gemini 2.5 Pro Exp Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_gemini25pro/raytraced_scene1.png" alt="Gemini 2.5 pro Exp - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_gemini25pro/raytraced_scene2.png" alt="Gemini 2.5 pro Exp - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_gemini25pro/raytraced_scene3.png" alt="Gemini 2.5 pro Exp - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_gemini25pro/raytraced_scene4.png" alt="Gemini 2.5 pro Exp - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Gemini 2.5 Pro 0506 Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_gemini25pro0506/raytraced_scene1.png" alt="Gemini 2.5 pro 0506 - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_gemini25pro0506/raytraced_scene2.png" alt="Gemini 2.5 pro 0506 - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_gemini25pro0506/raytraced_scene3.png" alt="Gemini 2.5 pro 0506 - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_gemini25pro0506/raytraced_scene4.png" alt="Gemini 2.5 pro 0506 - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Quasar Alpha Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_quasar_alpha/raytraced_scene1.png" alt="Grok 3 - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_quasar_alpha/raytraced_scene2.png" alt="Grok 3 - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_quasar_alpha/raytraced_scene3.png" alt="Grok 3 - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_quasar_alpha/raytraced_scene4.png" alt="Grok 3 - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Optimus Alpha Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_optimus_alpha/raytraced_scene1.png" alt="Optimus Alpha - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_optimus_alpha/raytraced_scene2.png" alt="Optimus Alpha - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_optimus_alpha/raytraced_scene3.png" alt="Optimus Alpha - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_optimus_alpha/raytraced_scene4.png" alt="Optimus Alpha - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Llama4 Scout Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_llama4_scout/raytraced_scene1.png" alt="llama4-scout - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_llama4_scout/raytraced_scene2.png" alt="llama4-scout - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_llama4_scout/raytraced_scene3.png" alt="llama4-scout - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_llama4_scout/raytraced_scene4.png" alt="llama4-scout - Test 4" width="400" />
    </td>
  </tr>
</table>

#### Llama4 Maverick Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_llama4_maverick/raytraced_scene1.png" alt="llama4-maverick - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_llama4_maverick/raytraced_scene2.png" alt="llama4-maverick - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_llama4_maverick/raytraced_scene3.png" alt="llama4-maverick - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_llama4_maverick/raytraced_scene4.png" alt="llama4-maverick - Test 4" width="400" />
    </td>
  </tr>
</table>

#### o4-Mini High Variance Test

<table align="center">
  <tr>
    <td align="center">
      <img src="variance_o4_mini_high/raytraced_scene1.png" alt="o4-Mini High - Test 1" width="400" />
    </td>
    <td align="center">
      <img src="variance_o4_mini_high/raytraced_scene2.png" alt="o4-Mini High - Test 2" width="400" />
    </td>
    <td align="center">
      <img src="variance_o4_mini_high/raytraced_scene3.png" alt="o4-Mini High - Test 3" width="400" />
    </td>
    <td align="center">
      <img src="variance_o4_mini_high/raytraced_scene4.png" alt="o4-Mini High - Test 4" width="400" />
    </td>
  </tr>
</table>

#### File sizes

There is a notable correlation between code creativity and file size.

<div align="center">
  <img src="python_file_sizes_by_llm.png" width="80%" />
</div>
