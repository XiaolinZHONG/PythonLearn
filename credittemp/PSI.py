


# dataAddress="F:/征信项目/train_sample_bin.csv"
from PythonLearnOne.ClassPythonTools import PythonTools
@PythonTools.timethis
def psi_check_fun(dataAddress,column_names):
    import pandas as pd
    import numpy as np
    reader = pd.read_csv(dataAddress, sep = '\t', iterator = True)
    chunks = []
    loop = True
    i = 1
    while loop:
        try:
            chunk = reader.get_chunk(3000000)
            chunks.append(chunk)
            print ( "正在读取数据块: " +str(i))
            i += 1
        except StopIteration:
            loop = False
            print ('数据读取完成')
    data = pd.concat(chunks, ignore_index = True)


    shape0=len(data)
    # s=data["hb_id"].groupby(data["htl_uidip_ord_ris1y"]).count()/shape0
    # test=pd.DataFrame(s)
    # tmp=pd.DataFrame()
    # test["col"]=test.index
    # test["rate"]=test["hb_id"]
    # print (test)
    #
    # tmp["col"]=test.index
    # tmp["rate"]=test.ix[1:,-1]

    # column_names=["c_uidcid_ord_cnt2y","c_uidip_amt_sum3m"]
    psi_df=pd.DataFrame()
    internal=pd.DataFrame({"":"","":""},index=[""])
    print("....")
    for col in column_names:
        tmp=pd.DataFrame({"rate":data["hb_id"].groupby(data[col]).count()/shape0}).reset_index()
        tmp.rename(columns={col:"bin"},inplace=True)
        index_num=tmp.shape[0]
        tmp.index=[col]*index_num   # import itertoools index=itertools.repeate("htl",3)
        # print(tmp)

        # a=list(np.array(tmp["bin"].ix[:-2]).ravel())
        # print(type(a))
        # print(a)
        # tmp["start"]=pd.Series(list(-900,tmp["bin"].ix[:-2]))
        psi_df=pd.concat([psi_df,internal,tmp],axis=0)
        # print(psi_df)
    #df1.groupby( [ "Name"] ).size().to_frame(name = 'count').reset_index()
    psi_df.to_csv("F:/征信数据/psi_bin_check.csv")
    return (print("完成"))



if __name__ == '__main__':

    dataAddress="F:/征信数据/tmp_credit_other_smooth_bin.txt"
    column_names=["complaint_ratio_ord_12m"
,"highamt_uid_ord_rat2y"
,"htl_star_uid_ord_rat1y"
,"htl_uid_newamt_msx2y"
,"htl_uid_ord_max2y"
,"htl_uidip_ord_ris1y"
,"lowamt_uid_ord_dmnsmax"
,"rcy_band_email"
,"uid_newamt_msx2y"
,"uid_ordtype_cnt2y"
,"train_bu_uid_ord_rat2y"
,"c_uidcid_ord_cnt2y"
,"c_uidip_amt_sum3m"
,"ns_uidcid_ord_rat2y"
,"ns_uidcid_ord_avg2y"
,"ns_uid_ord_dmnsmax"
,"highamt_uid_ord_ris1y"
,"flt_uidip_ord_cnt1y"
]
    psi_check_fun(dataAddress,column_names)