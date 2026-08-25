/**
 * DynamicForm 三级级联（scene → runtime_key → runtime_image）回归测试
 *
 * 覆盖用户要求的 9 个场景中与前端联动相关的部分：
 *  1/2 新增：scene 选择后 Runtime 下拉只显示该场景允许的 Runtime（联动配置来源 = 后端 column_related）
 *  3/4 新增：runtime_key 选择后 image 下拉只显示该 Runtime 已启用镜像
 *  5   编辑打开：formData 注入（init 信号）后立即按已有 scene/runtime_key 加载 options，不再 No Data
 *  6   编辑：当前引用镜像已停用仍保留显示（追加进 options 的历史值）
 *  7   编辑 scene 修改：runtime_key 与 runtime_image 都被清空，Runtime 下拉只剩新场景允许项
 *  8   编辑 runtime_key 修改：原 image 被清空，image options 只显示新 Runtime 镜像
 *
 * linkageConfig 结构与后端 _build_column_related() 输出一致（effectOption 键 = calculateId(dep 值)）。
 */
import React, { useEffect, useState } from "react";
import { render, fireEvent, act } from "@testing-library/react";
import { Form } from "antd";
import DynamicForm, {
  calculateId,
  ILinkageConfig,
  IDynamicFormConfigItem,
} from "../DynamicForm";

jest.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (k: string) => k, i18n: {} }),
}));

// antd 4 在 jsdom 下需要 matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: jest.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

const D1 = "msswift / 4.5.0-py311-cu128-r1 / harbor/mlops/msswift:4.5.0-py311-cu128-r1";
const D1B = "msswift / 4.5.0-py311-cu128-r2 / harbor/mlops/msswift:4.5.0-py311-cu128-r2";
const D2 = "msswift / 4.4.0-old / harbor/mlops/msswift:4.4.0-old"; // 已停用
const D3 = "gptqmodel / 7.1.0-py311-cu130-r2 / harbor/mlops/gptqmodel:7.1.0-py311-cu130-r2";
const D4 = "llama_factory / 4.1.0 / harbor/mlops/llama_factory:4.1.0";

// 与后端 _build_column_related() 一致的联动配置（enabled 镜像候选；D2 停用不在候选里）
const linkageConfig: ILinkageConfig[] = [
  {
    dep: ["scene"],
    effect: "runtime_key",
    effectOption: {
      [calculateId(["finetune"])]: ["msswift", "llama_factory", "deepspeed"].map((v) => ({
        label: v,
        value: v,
      })),
      [calculateId(["quantization"])]: ["gptqmodel"].map((v) => ({ label: v, value: v })),
    },
  },
  {
    dep: ["runtime_key"],
    effect: "runtime_image",
    effectOption: {
      [calculateId(["msswift"])]: [D1, D1B].map((v) => ({ label: v, value: v })),
      [calculateId(["gptqmodel"])]: [D3].map((v) => ({ label: v, value: v })),
      [calculateId(["llama_factory"])]: [D4].map((v) => ({ label: v, value: v })),
    },
  },
];

const mkConfig = (): IDynamicFormConfigItem[] => [
  {
    name: "scene",
    label: "场景",
    type: "select",
    required: true,
    data: {},
    rows: 0,
    options: [
      { label: "finetune", value: "finetune" },
      { label: "quantization", value: "quantization" },
    ],
  },
  {
    name: "runtime_key",
    label: "Runtime类型",
    type: "select",
    required: true,
    data: {},
    rows: 0,
    options: ["msswift", "llama_factory", "deepspeed", "gptqmodel"].map((v) => ({
      label: v,
      value: v,
    })),
  },
  {
    name: "runtime_image",
    label: "镜像版本",
    type: "select",
    required: true,
    data: {},
    rows: 0,
    options: [],
  },
];

// 模拟 ModalForm：form 实例 + formChangeRes state（onValuesChange / init 信号都走 setRes）
let formApi: {
  form: any;
  setRes: (res: any) => void;
} = null as any;

function Harness({ initial }: { initial?: Record<string, any> }) {
  const [form] = Form.useForm();
  const [res, setRes] = useState<any>(undefined);
  formApi = { form, setRes };
  // 与生产一致：config/configGroup 是稳定的 React state（ADUGTemplate 只 set 一次），
  // 不能每次渲染重建数组（否则 DynamicForm 初始化联动 effect 会因引用变化反复触发并清空已保留的值）
  const config = React.useMemo(() => mkConfig(), []);
  const configGroup = React.useMemo(() => [], []);
  // 模拟 ModalForm 编辑回填：setFieldsValue 不触发 onValuesChange
  useEffect(() => {
    if (initial) form.setFieldsValue(initial);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <Form form={form}>
      <DynamicForm
        form={form}
        primaryKey="id"
        config={config}
        configGroup={configGroup}
        linkageConfig={linkageConfig}
        formChangeRes={res}
      />
    </Form>
  );
}

const openSelect = (container: HTMLElement, index: number) => {
  const selector = container.querySelectorAll(".ant-select-selector")[index];
  fireEvent.mouseDown(selector);
};

const dropdownOptions = () =>
  Array.from(document.querySelectorAll(".ant-select-item-option-content")).map(
    (el) => el.textContent,
  );

const closeDropdown = () => fireEvent.keyDown(document.body, { key: "Escape" });

const settle = async () => {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 50));
  });
};

// init 信号（编辑回填注入；值已由 setFieldsValue 设置）
const emitInit = async (values: Record<string, any>) => {
  await act(async () => {
    formApi.setRes({ init: true, currentChange: values, allValues: values });
  });
  await settle();
};

// 用户主动修改（真实时序：antd 先把选择写入表单，再触发 onValuesChange → 非 init 信号）
const emitChange = async (change: Record<string, any>) => {
  await act(async () => {
    formApi.form.setFieldsValue(change);
    formApi.setRes({ currentChange: change, allValues: formApi.form.getFieldsValue() });
  });
  await settle();
};

describe("DynamicForm 三级级联", () => {
  afterEach(async () => {
    // 关闭残留下拉，避免跨用例干扰
    await act(async () => {
      document.body.innerHTML = "";
    });
  });

  it("T1 编辑打开：init 信号按已有 scene 加载 Runtime options 并保留值（不再 No Data）", async () => {
    const { container } = render(
      <Harness
        initial={{ scene: "finetune", runtime_key: "msswift", runtime_image: D1 }}
      />,
    );
    await settle();
    // 值注入后发出 init 信号（ModalForm 行为）
    await emitInit({ scene: "finetune", runtime_key: "msswift", runtime_image: D1 });
    expect(formApi.form.getFieldValue("scene")).toBe("finetune");
    expect(formApi.form.getFieldValue("runtime_key")).toBe("msswift"); // 值保留
    expect(formApi.form.getFieldValue("runtime_image")).toBe(D1); // 值保留
    openSelect(container, 1); // runtime_key 下拉
    await settle();
    const rtOptions = dropdownOptions();
    expect(rtOptions).toEqual(expect.arrayContaining(["msswift", "llama_factory", "deepspeed"]));
    expect(rtOptions).not.toContain("gptqmodel"); // 非 finetune Runtime 不出现
    closeDropdown();
    await settle();
    openSelect(container, 2); // image 下拉
    await settle();
    expect(dropdownOptions()).toEqual(expect.arrayContaining([D1, D1B]));
    expect(dropdownOptions()).not.toContain(D2); // 停用镜像不在新增候选里
    closeDropdown();
  });

  it("T2 编辑打开：当前引用镜像已停用仍保留显示（历史值追加进 options）", async () => {
    const { container } = render(
      <Harness
        initial={{ scene: "finetune", runtime_key: "msswift", runtime_image: D2 }}
      />,
    );
    await settle();
    await emitInit({ scene: "finetune", runtime_key: "msswift", runtime_image: D2 });
    expect(formApi.form.getFieldValue("runtime_image")).toBe(D2); // 停用镜像值保留
    openSelect(container, 2);
    await settle();
    const opts = dropdownOptions();
    expect(opts).toContain(D2); // 当前值追加显示
    expect(opts).toContain(D1); // enabled 镜像同时正常出现
    closeDropdown();
  });

  it("T3 新增：选 scene=finetune → Runtime 只显示 finetune 允许项；选 runtime_key=msswift → image 只显示 enabled msswift 镜像", async () => {
    const { container } = render(<Harness />);
    await settle();
    // 用户操作 scene（onValuesChange → 非 init 信号）
    await emitChange({ scene: "finetune" });
    openSelect(container, 1);
    await settle();
    const rtOptions = dropdownOptions();
    expect(rtOptions).toEqual(expect.arrayContaining(["msswift", "llama_factory", "deepspeed"]));
    expect(rtOptions).not.toContain("gptqmodel");
    closeDropdown();
    await settle();
    // 用户操作 runtime_key
    await emitChange({ runtime_key: "msswift" });
    openSelect(container, 2);
    await settle();
    const imgOptions = dropdownOptions();
    expect(imgOptions).toEqual(expect.arrayContaining([D1, D1B]));
    expect(imgOptions).not.toContain(D2); // 停用镜像不出现
    closeDropdown();
  });

  it("T4 新增：选 scene=quantization → Runtime 只剩 gptqmodel", async () => {
    const { container } = render(<Harness />);
    await settle();
    await emitChange({ scene: "quantization" });
    openSelect(container, 1);
    await settle();
    expect(dropdownOptions()).toEqual(["gptqmodel"]);
    closeDropdown();
  });

  it("T5 编辑修改 scene：runtime_key 与 runtime_image 级联清空，Runtime 下拉只剩新场景允许项", async () => {
    const { container } = render(
      <Harness
        initial={{ scene: "finetune", runtime_key: "msswift", runtime_image: D1 }}
      />,
    );
    await settle();
    await emitInit({ scene: "finetune", runtime_key: "msswift", runtime_image: D1 });
    // 用户把 scene 从 finetune 改为 quantization（非 init 信号）
    await emitChange({ scene: "quantization" });
    expect(formApi.form.getFieldValue("runtime_key")).toBeUndefined(); // 清空
    expect(formApi.form.getFieldValue("runtime_image")).toBeUndefined(); // 级联清空
    openSelect(container, 1);
    await settle();
    expect(dropdownOptions()).toEqual(["gptqmodel"]); // 只剩 quantization 允许项
    closeDropdown();
  });

  it("T6 编辑修改 runtime_key：原 image 清空，image options 只显示新 Runtime 镜像", async () => {
    const { container } = render(
      <Harness
        initial={{ scene: "finetune", runtime_key: "msswift", runtime_image: D1 }}
      />,
    );
    await settle();
    await emitInit({ scene: "finetune", runtime_key: "msswift", runtime_image: D1 });
    // 用户把 runtime_key 从 msswift 改为 llama_factory
    await emitChange({ runtime_key: "llama_factory" });
    expect(formApi.form.getFieldValue("runtime_image")).toBeUndefined(); // 原 msswift image 被清空
    openSelect(container, 2);
    await settle();
    const opts = dropdownOptions();
    expect(opts).toEqual([D4]); // 只显示 llama_factory 镜像
    expect(opts).not.toContain(D1); // 原 msswift 镜像不残留
    closeDropdown();
  });
});
