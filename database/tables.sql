drop table if exists ai_labs_user;
create table if not exists ai_labs_user(
    id integer primary key autoincrement,
    username text,
    fullname text,
    rolename text,
    gender text,
    mailaddr text,
    userpass text,
    aigc_temp_freq integer,
    aigc_perm_freq integer,
    date_time datetime
);
insert into ai_labs_user(username, fullname, rolename, gender, mailaddr, userpass, aigc_temp_freq, aigc_perm_freq, date_time) values ('guest', '游客', '买家', '男', 'guest@ai_labs.com', '084e0343a0486ff05530df6c705c8bb4', 3, 0, CURRENT_TIMESTAMP);
insert into ai_labs_user(username, fullname, rolename, gender, mailaddr, userpass, aigc_temp_freq, aigc_perm_freq, date_time) values ('ai_labs', 'AI-Labs', '卖家', '男', 'ai_labs@ai_labs.com', '35b569a2d1aafa6055450f9d1954ae67', 3, 0, CURRENT_TIMESTAMP);

drop table if exists ai_labs_chat;
create table if not exists ai_labs_chat(
    id integer primary key autoincrement,
    model_id text,
    username text,
    user text,
    assistant text,
    date_time datetime
);

drop table if exists ai_labs_images;
create table if not exists ai_labs_images(
    id integer primary key autoincrement,
    username text,
    user text,
    assistant text,
    date_time datetime
);

drop table if exists ai_labs_voice;
create table if not exists ai_labs_voice(
    id integer primary key autoincrement,
    username text,
    user_voice text,
    user_text text,
    assistant_voice text,
    assistant_text text,
    date_time datetime
);

-- 商品表，存储商品的基本信息
drop table if exists ai_labs_product_info;
create table if not exists ai_labs_product_info (
    id integer primary key autoincrement, -- 商品ID，自增主键
    category_id integer, -- 商品分类ID，外键关联分类表
    name text,           -- 商品名称
    title text,          -- 商品标题
    tags text,           -- 商品标签，多个标签用逗号分隔
    image text,          -- 商品平铺图
    video text,          -- 商品视频
    images text,         -- 商品图片，多张图片用逗号分隔
    gender text,         -- 男装、女装
    season text,         -- 季节分类
    price double,        -- 商品价格
    style text,          -- 风格
    material text,       -- 面料
    advantage text,      -- 亮点
    marketing text,      -- 营销文案
    description text,    -- 商品描述
    created_at datetime, -- 商品创建时间
    updated_at datetime  -- 商品更新时间
);

drop table if exists ai_labs_product_ratings;
create table if not exists ai_labs_product_ratings (
    id integer primary key autoincrement,
    user_id integer,
    product_id integer,
    rating integer,
    comment text,
    date_time datetime
);


drop table if exists ai_labs_service;
create table if not exists ai_labs_service(
    id integer primary key autoincrement,
    username text,
    user_voice text,
    user_text text,
    assistant_type text,
    assistant_text text,
    assistant_voice text,
    assistant_image text,
    assistant_product text,
    date_time datetime
);


drop table if exists ai_settings_gpt_api;
create table if not exists ai_settings_gpt_api(
    api_name text primary key,
    api_base text,
    api_key text,
    secret_key text,
    usage_scenario text,
    api_desc text
);

drop table if exists ai_settings_gpt_model;
create table if not exists ai_settings_gpt_model(
    api_name text,
    model_name text,
    model_url text,
    usage_scenario text,
    model_desc text,
    constraint pk_gpt_model primary key (api_name, model_name)
);
